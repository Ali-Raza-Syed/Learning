 k       = 5;
 n       = 2^k-1;
 [x,y,z] = sphere(n);
 surf(x,y,z);
 get(gcf, 'Renderer')
 hold on
 [X,Y,Z] = sphere(16);
 xx = 2*[0.5*X(:); 0.75*X(:); X(:)];
 yy = 2*[0.5*Y(:); 0.75*Y(:); Y(:)];
 zz = 2*[0.5*Z(:); 0.75*Z(:); Z(:)];
 scatter3(xx,yy,zz)
 get(gcf, 'Renderer')