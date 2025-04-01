function main(varargin)
    close all;
    
    % Retrieve still images
    imgs = {'rock_climber.jpg'};
    if (nargin > 0)
        for i = 1:nargin    
            imgs{i} = varargin{i};
        end
    end

    % Compile CPP for Pose Estimator
    setup;
    
    % Load Variables
    load("Calvin detector/pose_estimation_code_release_v1.22/code/env.mat");
    %load("Calvin detector/calvin_upperbody_detector_v1.04/code/detenv.mat");
    %UNNECESSARY - NOT USING CALVIN DETECTOR
    load ESRGAN_Release100/trained/ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat dlnG;
    
    baseImages = "Images/";
    superResolved = "super_resolved/";
    
    % Ensure directory for storing super-resolved images exists
    if ~isfolder(superResolved) 
        mkdir(superResolved);
    end
    
    % Construct file paths, extract names
    filePathNames = strcat(baseImages,imgs);
    fileNames = cell(numel(imgs));
    for i = 1:numel(filePathNames) 
        [~, name, ~] = fileparts(imgs{i});
        fileNames{i} = name;
    end
    
    % SUPER RESOLUTION STEP
    disp("Performing Super-Resolution");
    
    % Scaling factor, how much to scale up the image
    n = 2.0;
    exts = {'.jpg','.png','.tif'};
    
    testImages = imageDatastore(filePathNames,'FileExtensions',exts);
    
    figIdx = 1;
    for i = 1:numel(filePathNames)
        filePath = sprintf('%s/%s.png',superResolved,fileNames{i});
        if (isfile(filePath))
            orig = imread(sprintf("%s/%s",baseImages,imgs{i}));
            sr = imread(filePath);
    
            figure(figIdx);
            subplot(1,2,1);
            imshow(orig);
            title('Reference Image');
        
            subplot(1,2,2);
            imshow(sr);
            title('ESRGAN Super-Resolution');
        
            figIdx = figIdx + 1;
    
            continue;
        end
        % Future work, do object detection to find the human, draw bounding
        % box around him, get box coordinates, and crop to ROI for faster SR,
        % more efficient.
        % Read and preprocess the image
        Ireference = readimage(testImages, i);    
        IrefC = im2single(Ireference);
    
        % Super-resolve using ESRGAN
        IsisrC = ESRGAN_2xSuperResolution(IrefC); % IlowresC
    
        % Convert to uint8 (scale back to [0,255])
        IsisrC_uint8 = im2uint8(IsisrC);
        
        % filepath to save the super resolved image in
        output_filename = fullfile(superResolved, sprintf('%s.png', fileNames{i}));
        
        % Save the image with the new name
        imwrite(IsisrC_uint8, output_filename);
        
        figure(figIdx);
        subplot(1,2,1);
        imshow(IrefC);
        title('Reference Image');
    
        subplot(1,2,2);
        imshow(IsisrC);
        title('ESRGAN Super-Resolution');
    
        figIdx = figIdx + 1;
    end
    clear dlNg
    
    % SIFT CLUSTERING & POSE ESTIMATION
    
    for i = 1:numel(fileNames)
        % Load the image
        fname = sprintf("%s.png",fileNames{i});
        img = imread(sprintf("%s/%s",superResolved,fname));
        
        % Convert the image to grayscale if it's not already
        grayImg = rgb2gray(img);
        MinContrast = 0.02;   % Minimum contrast threshold (higher means fewer keypoints)
        NumOctaves = 7;       % Number of octaves for scale-space (higher value captures more scales)
        sigma = 10;
    
        disp("Performing SIFT Feature Detection");
        % Detect SIFT features with specified parameters
        points = detectSIFTFeatures(grayImg, "NumLayersInOctave",NumOctaves, "ContrastThreshold",MinContrast, Sigma= sigma);
        % Extract feature descriptors at the detected keypoints
        [~, validPoints] = extractFeatures(grayImg, points);
        
        figure(figIdx),
        subplot(1,3,1),
        title('SIFT Keypoints')
        imshow(img);
        hold on;
        plot(validPoints)
        hold off
        
        % Initialize clustering
        numKeypoints = validPoints.Count;
        clusterLabels = zeros(numKeypoints, 1); % Array to store cluster assignments
        numClusters = 0; % Counter for number of clusters
    
        disp("Performing SIFT Feature Clustering");
        for j = 1:numKeypoints
            if clusterLabels(j) == 0  % If the keypoint is not yet assigned
                % Look for similar clusters that are already assigned
                assignedCluster = 0;
                for k = 1:numKeypoints
                    if clusterLabels(k) ~= 0  % Check if a previous keypoint is assigned
                        similarity = sift_similarity(validPoints, j, k, sigma);
                        if similarity == 1  % If similar, take the assigned cluster label
                            clusterLabels(j) = clusterLabels(k);
                            assignedCluster = 1;  % Mark as assigned
                            break;  % Exit the loop once a similar cluster is found
                        end
                    end
                end
                
                
                % If no similar cluster was found, create a new cluster
                if assignedCluster == 0
                    numClusters = numClusters + 1;  % Create a new cluster
                    clusterLabels(j) = numClusters;
                    
                    % Assign similar keypoints to the same new cluster
                    for k = 1:numKeypoints
                        if clusterLabels(k) == 0  % If not already assigned
                            similarity = sift_similarity(validPoints, j, k,sigma);
                            if similarity == 1  % If similar, assign the same cluster
                                clusterLabels(k) = clusterLabels(j);
                            end
                        end
                    end
                end
            end
        end
        
        
        % Display the original image with clustered keypoints
        figure(figIdx),
        subplot(1,3,2),
        title('SIFT Clusters')
        imshow(img);
        hold on;
        colors = lines(numClusters);
        for j = 1:numKeypoints
            % Extract keypoint position, scale, and orientation
            x = validPoints.Location(i, 1); % X-coordinate of the i-th keypoint
            y = validPoints.Location(i, 2); % Y-coordinate of the i-th keypoint
            scale = validPoints.Scale(i);    % Scale of the i-th keypoint
            % Draw the keypoint as a circle
            viscircles([x, y], sigma*scale, 'EdgeColor', colors(clusterLabels(j),:), 'LineWidth', 0.1);
        end
        
        hold off;
        
        % Find the frequency of each cluster label
        [uniqueLabels, ~, labelIndices] = unique(clusterLabels); % Get unique labels and indices
        labelCounts = histcounts(labelIndices, length(uniqueLabels)); % Count occurrences of each unique label
        
        % Find the cluster label with the highest frequency
        [maxCount, maxIndex] = max(labelCounts);
        mostFrequentCluster = uniqueLabels(maxIndex);
        
        % Display the result
        disp(['The most frequent cluster label is: ', num2str(mostFrequentCluster)]);
        disp(['It appears ', num2str(maxCount), ' times.']);
        
        disp("Deriving Bounding Box");
        [imageHeight, imageWidth] = size(grayImg); 
        
        % Cluster label (example: we want the extremities for cluster 1)
        clusterId = mostFrequentCluster;
        
        % Find the indices of keypoints belonging to the specified cluster
        clusterIndices = find(clusterLabels == clusterId);
        
        % Get the coordinates (x, y) of the keypoints in the specified cluster
        xCoords = validPoints.Location(clusterIndices, 1);  % x-coordinates
        yCoords = validPoints.Location(clusterIndices, 2);  % y-coordinates
        %Radii = sigma * validPoints.Scale(clusterIndices);
        
        % Find the extremities (top, bottom, left, right)
        
        top = max(min(yCoords), 1);  % Ensure top does not go below 1
        bottom = min(max(yCoords), imageHeight);  % Ensure bottom does not exceed image height
        
        left = max(min(xCoords), 1);  % Ensure left does not go below 1
        right = min(max(xCoords), imageWidth);  % Ensure right does not exceed image width
        
        % Display the result
        disp(['Cluster ', num2str(clusterId), ' extremities:']);
        disp(['Top: ', num2str(top)]);
        disp(['Bottom: ', num2str(bottom)]);
        disp(['Left: ', num2str(left)]);
        disp(['Right: ', num2str(right)]);
        
        x = left; y = top; w = right - left; h = bottom - top;
        
        % MANUAL BB FOR IMAGE 000001.jpg
        %w = round(0.218947368421053 * imageWidth);
        %h = round(0.154411764705882 * imageHeight);
        %x = round(0.397426470588235 * imageWidth);
        %y = round(0.479473684210526 * imageHeight) - h;
        
        figure(figIdx),
        subplot(1,3,3),
        title('Bounding Box'),
        imshow(img),
        hold on;
        line([x,x+w],[y,y], "LineWidth", 5, 'Color','red');
        line([x,x+w],[y+h,y+h], "LineWidth", 5, 'Color','red');
        line([x,x],[y,y+h], "LineWidth", 5, 'Color','red');
        line([x+w,x+w],[y,y+h], "LineWidth", 5, 'Color','red');
        hold off;
        figIdx = figIdx + 1;
        disp("Performing Pose Estimation");
    
        bb = [x y w h]';
        parse_params_Buffy3and4andPascal.use_fg_high = false;
        PoseEstimStillImage(pwd, superResolved, fname, 1, 'full', bb, fghigh_params, parse_params_Buffy3and4andPascal, [], pm2segms_params, true);
        
        segmentIm = imread(sprintf('segms_full/%s',fname));
        figure(figIdx),
        imshow(segmentIm),
        title('Pose Estimated Image');
        figIdx = figIdx + 1;
    end
    
    clear all; % Ensure the CPP code is shut down (it's a bit memory leaky)
end