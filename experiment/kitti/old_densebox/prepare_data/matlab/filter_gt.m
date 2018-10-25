function [all_out_anno] = filter_gt(all_img_anno, min_height, max_occlusion, max_truncation, ignore_van)
%FILTER_GT
% Remove instances of all types except for 'Van' and 'Car'
%   For KITTI hard setting, min_height = 25,  max_occlusion = 2  max_truncation = 0.5
%
% Input
%   all_img_anno  -  original annotation
%
% Output
%   all_out_anno  -  new annotation

all_out_anno.label_name = all_img_anno.label_name;
all_out_anno.label_count = all_img_anno.label_count;
all_out_anno.images = {};
all_label_name = all_out_anno.label_name;

% each image
n_img = length(all_img_anno.images);
for id = 1 : n_img
    img_anno = all_img_anno.images{id};

    out_anno.name = img_anno.name;
    out_anno.instance = {};
    out_anno.bbox = {};
    out_anno.ignore = {};
    out_anno.label_id = {};
    out_anno.occlusion = {};
    out_anno.truncation = {};
    ignore_flag = zeros(1, length(img_anno.instance));

    for i = 1 : length(img_anno.instance)
        if img_anno.ignore{i} == 1
            ignore_flag(i) = 1;
            continue;
        end

        cur_label_id = img_anno.label_id{i};
        cur_label_name = all_label_name{cur_label_id};
        if ignore_van == true
            if strcmpi(cur_label_name, 'Van') == true
                img_anno.ignore{i} = 1;
                continue;
            end
        end

        %%
        if(strcmpi(cur_label_name, 'DontCare') == true)
            ignore_flag(i) = 1;
            img_anno.ignore{i} = 1;
            continue;
        end

        if strcmpi(cur_label_name, 'Van') == false ...
                && strcmpi(cur_label_name, 'Car') == false ...
                && strcmpi(cur_label_name, 'DontCare') == false
            ignore_flag(i) = 1;
            continue;
        end

        bbox = img_anno.bbox{i};
        height = bbox(4) - bbox(2);

        if height < min_height || img_anno.occlusion{i} > max_occlusion || img_anno.truncation{i} > max_truncation
            img_anno.ignore{i} = 1;
            continue;
        end
    end

    for i = 1 : length(img_anno.instance)
        if ignore_flag(i) == 0
            out_anno.instance{end + 1} = img_anno.instance{i};
            out_anno.bbox{end + 1} = img_anno.bbox{i};

            out_anno.ignore{end + 1} = img_anno.ignore{i};
            out_anno.label_id{end + 1} = img_anno.label_id{i};

            out_anno.occlusion{end + 1} = img_anno.occlusion{i};
            out_anno.truncation{end + 1} = img_anno.truncation{i};
        end
    end
    all_out_anno.images{end + 1} = out_anno;
end
