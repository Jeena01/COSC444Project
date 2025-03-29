function similarity = sift_similarity(points, i, j, sigma)
    x_i = points.Location(i,1);
    y_i =  points.Location(i,2);
    x_j = points.Location(j,1);
    y_j = points.Location(j,2);

    scale_i = points.Scale(i);
    scale_j = points.Scale(j);

    distance = sqrt((y_i-y_j)^2+(x_i-x_j)^2);

    if distance < sigma*(scale_i+scale_j)
        similarity = 1;
    else
        similarity =0;
    end
end