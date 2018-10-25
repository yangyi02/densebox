% Convert original label stored in 'objectKITTITrain.mat' to the new
% format and save to 'all_structured_anno.mat'.
%
% 'objectKITTITrain.mat' contains a 7481-long cell array named as 'objects'
% Each objects{i} is a struct with the following structure:
%          type: 'Pedestrian'
%    truncation: 0
%     occlusion: 0
%         alpha: -0.2000
%            x1: 712.4000
%            y1: 143
%            x2: 810.7300
%            y2: 307.9200
%             h: 1.8900
%             w: 0.4800
%             l: 1.2000
%             t: [1.8400 1.4700 8.4100]
%            ry: 0.0100
%
% In the generated 'all_structured_anno.mat', there is a struct called 'anno', which
% has three elements
%         images: {1x7481 cell}
%     label_name: {1x9 cell}
%    label_count: {1x9 cell}
%
% Each images{i} has something like:
% name: '000000'
%      instance: {[761.5650 225.4600 1]}
%          bbox: {[4x1 double]}
%     occlusion: {[0]}
%        ignore: {[0]}
%      label_id: {[1]}
%    truncation: {[0]}
%
% label_name{1} is: 'Pedestrian'
% label_count{1} is: 4487

% load original data
mat = load('mat/objectKITTITrain.mat');
objects = mat.objects;

% dimension
n_img = length(objects);

label_name = {};
label_count = zeros(1, 10);
anno = {};
anno.images = {};
anno.label_name = {};
anno.label_count = {};

% each image
for i = 1 : n_img
    % print
    if (mod(i, 500) == 0)
       fprintf('%d/%d\n', i, n_img);
    end
    anno.images{end + 1} = {};
    name = sprintf('%06d', i - 1);

    % create a node
    anno.images{end}.name = name;
    anno.images{end}.instance = {};
    anno.images{end}.bbox = {};
    anno.images{end}.occlusion = {};
    anno.images{end}.ignore = {};
    anno.images{end}.label_id = {};
    anno.images{end}.truncation = {};

    % each instance labeled for each image
    cur_instances = objects{i};
    for instance_id = 1 : length(cur_instances)

        instance = cur_instances(instance_id);

        % add bbox
        temp_name = instance.type;
        bbox = zeros(4, 1);
        bbox(1) = instance.x1;
        bbox(2) = instance.y1;
        bbox(3) = instance.x2 ;
        bbox(4) = instance.y2 ;
        anno.images{end}.bbox{end + 1} = bbox;

        % center
        point = ones(1, 3);
        point(:, 1 : 2) = point(:, 1 : 2) * -1;
        point(1, 1) = (bbox(1) + bbox(3)) / 2;
        point(1, 2) = (bbox(2) + bbox(4)) / 2;
        anno.images{end}.instance{end + 1} = point;

        % other
        anno.images{end}.occlusion{end + 1} = instance.occlusion;
        anno.images{end}.truncation{end + 1} = instance.truncation;
        anno.images{end}.ignore{end + 1} = 0;

        % for statics
        label_id = -1;
        for i = 1:length(anno.label_name)
            if strcmp(temp_name, anno.label_name{i}) == 1
                label_id = i;
                break;
            end
        end

        if label_id == -1
           anno.label_name{end + 1} = temp_name;
           anno.label_count{end + 1} = 0;
           label_id = length(anno.label_name);
        end
        anno.images{end}.label_id{end + 1} = label_id;

        anno.label_count{label_id} = anno.label_count{label_id} + 1;
    end
end

% save result
save('mat/all_structured_anno.mat', 'anno');
