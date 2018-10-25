function varargout = plotbox(b,varargin)
% Plot bounding box.

nbox = size(b,1);

hs = [];
hold on
for i=1:nbox
    h = plot([b(i,1) b(i,3) b(i,3) b(i,1) b(i,1)], ...
             [b(i,2) b(i,2) b(i,4) b(i,4) b(i,2)], ...
             varargin{:});
    hs = [hs; h];
end

if nargout == 1
    varargout{:} = hs;
end
