% Pre-process image list.

% img_fold = '/data/KITTI/training/image_2/';
img_fold = '~/data/kitti/data_object_image_2/training/image_2/';
annotation_fold = './annotations/';
out_fold = './train_jpg/';
%lr_out_fold = './lr_train_jpg/';

out_filename_prefix = 'kitti/';

%% preprocess
mat = load('mat/all_structured_anno.mat');
anno0 = mat.anno;

% For KITTI hard setting
%   min_height = 25
%   max_occlusion = 2
%   max_truncation = 0.5
anno = filter_gt(anno0, 25, 2, 0.5, true);
save('mat/all_structured_anno_gt.mat', 'anno')

%% validation id
n_img = length(anno.images);
is_val = rand(n_img, 1) > 0.95;
save('mat/val_idx.mat', 'is_val');
mat = load('mat/val_idx.mat');
is_val = mat.is_val;
format shortG

%% for visualization
color = {'mo','yo','bo','go','ro','co','mo','yo','b.','g.','r.','c.','m.','y.','b*','g*','r*','c*','m*','y*'};

%% file handler
train_fid = fopen([annotation_fold, 'new_original_train_jpg.txt'], 'w');
val_fid = fopen([annotation_fold, 'new_original_val_jpg.txt'], 'w');
fddb_val_fid = fopen([annotation_fold, 'new_original_val_fddb.txt'], 'w');
val_filename_fid = fopen([annotation_fold, 'val_filename.txt'], 'w');

n_points = 0;
flip_pairs = [];
img_size_h = -1;
img_size_w = -1;

write_img = true;

%% each image
for i = 1 : n_img
    % progress
    if mod(i, 100) == 0
        fprintf('%d/%d\n', i, n_img);
    end

    % original image
    full_file_name = [img_fold,anno.images{i}.name '.png'];
    S = regexp(anno.images{i}.name, '\.', 'split');
    file_name = S{1};
    file_name = [out_filename_prefix, file_name];
    n_instance = length(anno.images{i}.instance);

    % load image
    img = imread(full_file_name);
    img_size_h = size(img, 1);
    img_size_w = size(img, 2);

    % load instances
    all_instance = anno.images{i}.instance;
    ignore = anno.images{i}.ignore;
    other_bboxes = anno.images{i}.bbox ;
    for inst_id = 1 : n_instance
        other_bboxes{inst_id} = other_bboxes{inst_id}';
    end

    % used for validation
    if is_val(i) > 0
        fprintf(val_filename_fid, '%s \n', ['', file_name]);
        fprintf(fddb_val_fid, '%s \n', ['', file_name]);
        fprintf(fddb_val_fid, '%d \n', n_instance);
        for instance_id = 1 : n_instance
            bbox = other_bboxes{instance_id};
            bbox = bbox';
            if(ignore{instance_id} == 0)
                fprintf(fddb_val_fid, '%.2f %.2f %.2f %.2f 0 \n', bbox(1), bbox(2), bbox(3) - bbox(1), bbox(4) - bbox(2));
            else
                fprintf(fddb_val_fid, '%.2f %.2f %.2f %.2f 1 \n', bbox(1), bbox(2), bbox(3) - bbox(1), bbox(4) - bbox(2));
            end
        end
    end

    % write cropped image
    if write_img == true
        imwrite(img, [out_fold,'', file_name , '.jpg'], 'jpg', 'Quality', 100);
    end
    cur_fid = [];
    if is_val(i) == 0
        cur_fid = train_fid;
    else
        cur_fid = val_fid;
    end
    fprintf(cur_fid, '%s ', ['', file_name]);

    % each instance
    for inst_id = 1 : n_instance
        other_bbox = other_bboxes{inst_id};
        for p_id = 1 : n_points
            if points(p_id, 3) == 0
                fprintf(cur_fid, '%.2f %.2f ', points(p_id, 1), points(p_id,2));
            else
                fprintf(cur_fid, '-1 -1 ');
            end

        end
        for p_id = 1 : n_points
            if points(p_id, 3) == 1
                fprintf(cur_fid, '%.2f %.2f ',points(p_id,1),points(p_id,2));
            else
                fprintf(cur_fid, '-1 -1 ');
            end
        end

        % 16 points + 5 + 1 +2 = 24
        fprintf(cur_fid, '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ', ...
                other_bbox(1), other_bbox(2), ...
                other_bbox(3), other_bbox(2), ...
                other_bbox(3), other_bbox(4), other_bbox(1), other_bbox(4), ...
                other_bbox(1)/2 + other_bbox(3)/2, other_bbox(2)/2 + other_bbox(4)/2);
        fprintf(cur_fid,'%.2f %.2f ',ignore{inst_id} ,ignore{inst_id} );
    end
    fprintf(cur_fid,'\n' );

    %% show for debug
    cla, imagesc(img), axis image, hold on
    for inst_id = 1 : n_instance
        other_bbox = other_bboxes{inst_id};
        if(ignore{inst_id} == 0)
            plotbox(other_bbox, 'g--');
        else
            plotbox(other_bbox, 'r--');
        end
    end
    for inst_id = 1 : n_instance
        points = all_instance{inst_id};
        for p_id = 1:n_points
            if points(p_id, 3) == 1
                plot(points(p_id, 1), points(p_id, 2), 'ro');
                text(points(p_id, 1), points(p_id, 2), num2str(p_id), 'Color', 'red', 'FontSize', 15);
            else
                plot(points(p_id, 1), points(p_id, 2), 'go');
                text(points(p_id, 1), points(p_id, 2), num2str(p_id), 'Color', 'red', 'FontSize', 15);
            end
        end
    end

    %% left right flip
    if write_img == true
        cropped_img = img;
        cropped_img(:, :, 1) = fliplr(img(:, :, 1));
        cropped_img(:, :, 2) = fliplr(img(:, :, 2));
        cropped_img(:, :, 3) = fliplr(img(:, :, 3));
    end

    for inst_id = 1 : n_instance
        points = all_instance{inst_id};
        other_bbox = other_bboxes{inst_id};
        other_bbox = [img_size_w - other_bbox(1), other_bbox(2), img_size_w - other_bbox(3), other_bbox(4)];
        other_bbox = [other_bbox(3), other_bbox(2), other_bbox(1), other_bbox(4)];

        for point_id = 1 : n_points
            if points(point_id,1) == -1 || points( point_id,2) == -1
                continue;
            end
            points( point_id,1) = img_size_w - points(point_id,1) ;
        end
        for flip_id = 1 : size(flip_pairs, 1)
            temp = points(flip_pairs(flip_id,1),:);
            points(flip_pairs(flip_id,1),:) = points( flip_pairs(flip_id,2),:);
            points(flip_pairs(flip_id,2),:) = temp;
        end
        all_instance{inst_id} = points;
        other_bboxes{inst_id} = other_bbox;
    end

    %% write flipped image
    if write_img == true
        imwrite(cropped_img, [out_fold, file_name,'_lr', '.jpg'], 'jpg', 'Quality', 100);
    end
    cur_fid = [];
    if is_val(i) == 0
        cur_fid = train_fid;
    else
        cur_fid = val_fid;
    end
    fprintf(cur_fid, '%s ', [file_name,'_lr']);

    for inst_id = 1 : n_instance
        points = all_instance{inst_id};

        other_bbox = other_bboxes{inst_id};
        for p_id = 1: n_points
            if points(p_id, 3) == 0
                fprintf(cur_fid, '%.2f %.2f ', points(p_id,1), points(p_id,2));
            else
                fprintf(cur_fid, '-1 -1 ' );
            end
        end
        for p_id = 1 : n_points
            if points(p_id, 3) == 1
                fprintf(cur_fid, '%.2f %.2f ',points(p_id,1),points(p_id,2));
            else
                fprintf(cur_fid, '-1 -1 ');
            end
        end
        fprintf(cur_fid,'%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ', ...
                other_bbox(1), other_bbox(2) ,...
                other_bbox(3), other_bbox(2), other_bbox(3), other_bbox(4), other_bbox(1), other_bbox(4), ...
                other_bbox(1)/2 + other_bbox(3)/2, other_bbox(2)/2 + other_bbox(4)/2);
        fprintf(cur_fid,'%.2f %.2f ',ignore{inst_id} ,ignore{inst_id} );
    end
    fprintf(cur_fid, '\n');

    % show for debug
    % cropped_img = img;
    % cropped_img(:,:,1) = fliplr(img(:,:,1));
    % cropped_img(:,:,2) = fliplr(img(:,:,2));
    % cropped_img(:,:,3) = fliplr(img(:,:,3));
    % cla, imagesc(cropped_img), axis image, hold on

    % for inst_id = 1:n_instance
    %     other_bbox =  other_bboxes{inst_id};
    %     if(ignore{inst_id} == 0)
    %         plotbox(other_bbox,'g--');
    %     else
    %         plotbox(other_bbox,'r--');
    %     end
    % end

    % for inst_id = 1:n_instance
    %     points = all_instance{inst_id};

    %     for p_id = 1:n_points
    %         if points(p_id,3) == 1
    %             plot(points(p_id,1),points(p_id,2),'ro');
    %             text(points(p_id,1),points(p_id,2) ,num2str(p_id),'Color','red','FontSize',15);
    %         else
    %             plot(points(p_id,1),points(p_id,2),'go');
    %             text(points(p_id,1),points(p_id,2) ,num2str(p_id),'Color','red','FontSize',15);
    %         end
    %     end
    % end
end

fclose(train_fid);
fclose(val_fid);
fclose(fddb_val_fid);
fclose(val_filename_fid);
