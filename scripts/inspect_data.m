close all; clear all;

TRAIN_DIR = '/home/rolledm1/Desktop/darknet-train-11';

% define classes (out of the 200 ILSVRC detection classes) of interest
classes = {'apple', 'bowl', 'cup or mug', 'iPod', 'lemon', 'lipstick', 'orange', ...
           'remote control', 'saltor pepper shaker', 'water bottle', 'wine bottle'};
       
N_SHOW    = 15; % how many images to check out

for ii=1:N_SHOW
    class_i   = randi(length(classes));
    class     = strrep(classes{class_i}, ' ', '');
    img_dir   = [TRAIN_DIR '/images/' class '/'];
    dir_list  = dir(fullfile(img_dir, '*.JPEG'));
    img_i     = randi(length(dir_list));
    img_name  = dir_list(img_i).name;
    img       = imread([img_dir img_name]);
    splits    = strsplit(img_name, '.');
    label_f   = [TRAIN_DIR '/labels/' class '/' splits{1} '.txt'];
    label_fid = fopen(label_f);
    labels    = textscan(label_fid, '%f %f %f %f %f');
    fclose(label_fid);
    labels    = cell2mat(labels);
    
    figure; imshow(img); hold on;
    
    for jj=1:size(labels, 1)
        rectangle('Position', ...
                  [labels(jj, 2)*size(img, 2)+1 ...
                   labels(jj, 3)*size(img, 1)+1 ...
                   labels(jj, 4)*size(img, 2)   ...
                   labels(jj, 5)*size(img, 1)], 'EdgeColor', 'r', 'LineWidth', 2);
        text(labels(jj, 2)*size(img, 2)+5, ...
             labels(jj, 3)*size(img, 1)+10, ...
             class, 'Color', 'red', 'FontSize', 14);
    end
    
end