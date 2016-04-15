% Gets images/bounding boxes from ILSVRC2014 detection dataset and puts the
% images/labels in the required directory structure for fine-tuning the
% YOLO CNN.  See here for more info: https://github.com/Guanghan/darknet

% setup in input/output directories
DEV_KIT_ROOT  = '/home/rolledm1/Desktop/ILSVRC2014_devkit';
BBOX_ROOT     = '/media/rolledm1/rolledm1-ext1/ILSVRC2014_DET_bbox_train';
IMAGE_ROOT    = '/media/rolledm1/rolledm1-ext1/ILSVRC2014_DET_train';
IMAGE_EXT     = '.JPEG';
OUT_DIR       = '/media/rolledm1/rolledm1-ext1/darknet-train';
OUT_IMAGE_DIR = [OUT_DIR '/images'];
OUT_LABEL_DIR = [OUT_DIR '/labels'];

% define classes (out of the 200 ILSVRC detection classes) of interest
classes = {'apple', 'bowl', 'cup or mug', 'iPod', 'lemon', 'lipstick', 'orange', ...
           'remote control', 'salt or pepper shaker', 'water bottle', 'wine bottle'};

if ~isdir(OUT_DIR)
    mkdir(OUT_DIR);
end
if ~isdir(OUT_IMAGE_DIR)
    mkdir(OUT_IMAGE_DIR);
end
if ~isdir(OUT_LABEL_DIR)
    mkdir(OUT_LABEL_DIR);
end

% load metadata: synsets(1:200) struct array contains 200 detection classes
load([DEV_KIT_ROOT '/data/meta_det.mat']);

class_structs = [];
cs_i = 1;

for ii=1:200
   if ~isempty(strmatch(synsets(ii).name, classes, 'exact'))
       disp(['Found match: ' synsets(ii).name]);
       s.det_id = ii;
       s.wnid   = synsets(ii).WNID;
       s.name   = strrep(synsets(ii).name, ' ', '');
       s.bboxes = {}; % array of bounding box matrices (1 matrix for each image)
       
       % read list of img files associated with this class
       img_list_file = [DEV_KIT_ROOT '/data/det_lists/train_pos_' num2str(ii) '.txt'];
       fid = fopen(img_list_file);
       s.img_list = textscan(fid, '%s');
       fclose(fid);
       s.img_list = s.img_list{1};
       
       % for all images, get corresponding bounding boxes for this class
       for jj=1:length(s.img_list)
           xml_file  = [BBOX_ROOT '/' s.img_list{jj} '.xml'];
           bb_struct = VOCreadxml(xml_file);
           
           % add image extension
           s.img_list{jj} = [s.img_list{jj} IMAGE_EXT];
           
           img_w = str2num(bb_struct.annotation.size.width);
           img_h = str2num(bb_struct.annotation.size.height);
           objs  = bb_struct.annotation.object;
           
           img_bbs = [];
           
           for kk=1:length(objs)
               % make sure we're grabbing the correct object's bb
               if strcmp(objs(kk).name, s.wnid)
                   bb_xmin = str2num(objs(kk).bndbox.xmin);
                   bb_xmax = str2num(objs(kk).bndbox.xmax);
                   bb_ymin = str2num(objs(kk).bndbox.ymin);
                   bb_ymax = str2num(objs(kk).bndbox.ymax);
                   
                   bb_yolo = [bb_xmin/img_w, bb_ymin/img_h, ...
                              (bb_xmax-bb_xmin)/img_w, ...
                              (bb_ymax-bb_ymin)/img_h];
                   img_bbs = [img_bbs; bb_yolo];
               end
           end
           s.bboxes = [s.bboxes; img_bbs];
       end
       class_structs = [class_structs, s];
   else
      disp(['Skipping ' synsets(ii).name '...']);
   end
end

disp('Done getting boxes & images...');

tlist_fid = fopen([OUT_DIR '/training_list.txt'], 'w');
if tlist_fid < 0
    error('Failed to open training list file.');
end

for ii=1:length(class_structs)
    s = class_structs(ii);
    
    disp(['processing ' s.name ', (' num2str(ii) '/' num2str(length(class_structs)) ')']);
        
    c_image_dir = [OUT_IMAGE_DIR '/' s.name];
    c_label_dir = [OUT_LABEL_DIR '/' s.name];
    mkdir(c_image_dir);
    mkdir(c_label_dir);
    
    for jj=1:length(s.img_list)
        strs = strsplit(s.img_list{jj}, '/');
        
        img_name = strs{2};
        l = strsplit(img_name, '.');
        label_name = l{1};
        
        % copy image to output directory
        [success, msg, msgid] = copyfile([IMAGE_ROOT '/' s.img_list{jj}], ...
                                         [c_image_dir '/' img_name]);
        if ~success
            error(['Error copying file: ' msg]);
        end                                     
                                     
        fprintf(tlist_fid, '%s\n', [c_image_dir '/' img_name]);
        
        % write a text file of bounding boxes for this image
        fid = fopen([c_label_dir '/' label_name '.txt'], 'w');
        
        if fid < 0
            error('Failed to open label file.');
        end
        
        for kk=1:size(s.bboxes{jj}, 1)
            fprintf(fid, '%d %f %f %f %f', ii-1, ...
                    s.bboxes{jj}(kk, 1), ...
                    s.bboxes{jj}(kk, 2), ...
                    s.bboxes{jj}(kk, 3), ...
                    s.bboxes{jj}(kk, 4));
            
            if kk < size(s.bboxes{jj}, 1)
                fprintf(fid, '\n');
            end
        end
        fclose(fid);
    end
end
fclose(tlist_fid);
disp('Done.');