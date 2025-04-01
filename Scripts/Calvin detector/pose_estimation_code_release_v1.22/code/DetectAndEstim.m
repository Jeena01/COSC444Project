function DetectAndEstim(img_dir,img_name,pffubfmodel_path,facemodel_path,det_pars,classname,fghigh_pars,parse_pars,addinf_pars,segm_pars,verbose)
    %parse_pars.use_fg_high = false % uncomment this line if you want to skip the foreground highlighting stage
    stick_coor = cell(0);
    T = struct('D',{}, 'FGH',{}, 'PM',{},'CM',{});
    
    outname_mat = fullfile(img_dir,[img_name '_' classname '_pms.mat']);
    if exist(outname_mat,'file')
      return;
    end
    if verbose
    t = tic;
    end
    detections = DetectStillImage(fullfile(img_dir,img_name),pffubfmodel_path,facemodel_path,det_pars,verbose);
    if verbose 
    disp(['Detecting people: ' num2str(toc(t)) ' sec.']);
    end
    if ~isempty(detections)
    %convert coordinates from [x1 y1 x2 y2] to [x y width height]
    detections(:,3:4) = detections(:,3:4) - detections(:,1:2) +1;
    
    temp_dir = 'temp';
    if ~exist(fullfile(img_dir,temp_dir),'dir')
      mkdir(fullfile(img_dir,temp_dir));
    end
    dets_dir = 'dets';
    if ~exist(fullfile(img_dir,dets_dir),'dir')
      mkdir(fullfile(img_dir,dets_dir));
    end
    
    img = imread(fullfile(img_dir,img_name));
    
    for dix=1:size(detections,1) % run pose estimation for every detection
      [T(dix) stick_coor{dix}] = PoseEstimStillImage(pwd,'/Images',img_name,dix, classname, round(detections(dix,1:4)'), fghigh_pars, parse_pars, addinf_pars, segm_pars, verbose);      
    end
    
    if ~isempty(detections)
      for d=1:size(detections,1)
        img = PaintBB(img,round(detections(d,1:4)),[1 0 0],[1 2 3]);
        figure(d),
        imshow(img);
      end
    end
    save(outname_mat,'T','stick_coor','detections');
end