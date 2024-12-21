x6 = ReadMyImage('TargetImage.bmp');
DisplayMyImage(x6)
title('Football Team');
h6 = ReadMyImage('FILTER.bmp');
DisplayMyImage(h6)
title('Face Filter');
y1 = DSLSI2D(h6,x6);
DisplayMyImage(abs(y1))
title('|y[m,n]|');
s1 = abs(y1).^4;
DisplayMyImage(s1)
title('|y[m,n]|^4');
s2 = abs(y1).^6;
DisplayMyImage(s2)
title('|y[m,n]|^6');
function [x] = ReadMyImage(string)
x=double((rgb2gray(imread(string))));
x=x-min(min(x));
x=x/max(max(x));
x=x-0.5;
end
function []=DisplayMyImage(Image)
Image=Image-min(min(Image));
figure;
imshow(uint8(255*Image/max(max(abs(Image)))));
end
function[y]=DSLSI2D(h,x)
[hx, hy] = size(h);
[xx, xy] = size(x);
yx = hx+xx-1 ;
yy = hy+xy-1 ;
y = zeros(yx, yy);
h = rotate180(h);
xa = zeros(2*(hx-1)+xx, 2*(hy-1)+xy);
xa(hx:yx,hy:yy) = x ;
for i = 1:yy
for j = 1:yx
s = xa(j:(j-1)+hx,i:(i-1)+hy).*h;
y(j,i) = summation(s);
end
end
end
%rotate function
function [ y ] = rotate180( h )
    [hx, hy] = size(h);
    x = h ;
    y = h ;
    for i = 1:hx
        x(i,:) = h(hx-(i-1),:);
    end 
    for i = 1:hy
        y(:,i) = x(:,hy-(i-1));
    end 
end
%summation function
function [ y ] = summation( h )
[hx, hy] = size(h);
y = 0 ;
for i = 1:hy
for j = 1:hx
y = y + h(j,i);
end
end
end
