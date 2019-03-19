% two inputs with 3 first layer nodes with thresholding and ReLU activation

W = ( -1 + (1-(-1)).*rand(2, 5)* 1e1 );

b = ( -1 + (1-(-1)).*rand(5, 1)* 1e1 );

% w1 = 2e-1;
% w2 = 3e-1;
% b1 = 1e-1;
% w3 = -3e-1;
% w4 = -2e-1;
% b2 = 10e-1;
% w5 = -5e-1;
% w6 = 1e-1;
% b3 = 10e-1;

learning_rate = 1e-2;
x1 = -1:0.02:1;
x2 = -1:0.02:1;

threshold = 1;

p_x1 = 0 + (1-0).*rand(1, 20);
p_x2 = 0 + (1-0).*rand(1, 20);
p_z = 0 + (1-0).*rand(1, 20);

P = [p_x1; p_x2];

dW = zeros([2, 3]);
db = zeros([3, 1]);

% dw1 = 0;
% dw2 = 0;
% dw3 = 0;
% dw4 = 0;
% dw5 = 0;
% dw6 = 0;
% db1 = 0;
% db2 = 0;
% db3 = 0;

[X1,X2] = meshgrid(x1, x2);
Z = zeros(size(X1));
for i = 1 : size(W, 2)
    Z = Z + X1.*W(1, i) + X2.*W(2, i) + b(i);
    Z(Z < 0) = 0;
end

% Z1 = X1.*W(1, 1) + X2.*W(2, 1) + b(1);
% Z2 = X1.*W(1, 2) + X2.*W(2, 2) + b(2);
% Z3 = X1.*W(1, 3) + X2.*W(2, 3) + b(3);
% 
% Z1(Z1 < 0) = 0;
% Z2(Z2 < 0) = 0;
% Z3(Z3 < 0) = 0;
% 
% A1 = Z1;
% A2 = Z2;
% A3 = Z3;
% 
% Y_cap = A1 + A2 + A3;
% [dfdx,dfdy] = gradient(Y_cap);

scatter_handle = scatter3(P(1, :), P(2, :), p_z, 'black');
hold on
surf_handle = surf(X1, X2, Z);
colormap(gca(), lines(8));
surf_handle.FaceAlpha = 0.5;
axis([-1, 1, -1, 1, -1, 1]);
while true
    
    dW = zeros(size(W));
    db = zeros([size(W, 2), 1]);
    
%     dw1 = 0;
%     dw2 = 0;
%     dw3 = 0;
%     dw4 = 0;
%     dw5 = 0;
%     dw6 = 0;
%     db1 = 0;
%     db2 = 0;
%     db3 = 0;

    J = 0;
    for i = 1:20
        
        Z = W' * P(:, i) + b;
        
%         z1 = p_x1(i)*w1 + p_x2(i)*w2 + b1;
%         z2 = p_x1(i)*w3 + p_x2(i)*w4 + b2;
%         z3 = p_x1(i)*w5 + p_x2(i)*w6 + b3;
        
        Z(Z < 0) = 0;
        
%         z1(z1 < 0) = 0;
%         z2(z2 < 0) = 0;
%         z3(z3 < 0) = 0;
        A = Z;
        
%         a1 = z1;
%         a2 = z2;
%         a3 = z3;
        
        Y_cap = sum(A);

%         y_cap = a1 + a2 + a3;

        update_flags = Z > 0;
        
%         update_flag_z1 = z1 > 0;
%         update_flag_z2 = z2 > 0;
%         update_flag_z3 = z3 > 0;
        
        J = J + (Y_cap - p_z(i))^2;
        
        points = repmat([p_x1(i);
                         p_x2(i)], 1, size(W, 2));

        dW = dW + 2*(Y_cap - p_z(i)) .* points .* repmat(update_flags', size(W, 1), 1);
        
%         dw1 = dw1 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z1;
%         dw2 = dw2 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z1;
%         dw3 = dw3 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z2;
%         dw4 = dw4 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z2;
%         dw5 = dw5 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z3;
%         dw6 = dw6 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z3;

        db = db + 2 .* (Y_cap - p_z(i)) .* update_flags;

%         db1 = db1 + 2*(y_cap - p_z(i))*update_flag_z1;
%         db2 = db2 + 2*(y_cap - p_z(i))*update_flag_z2;
%         db3 = db3 + 2*(y_cap - p_z(i))*update_flag_z3;
        
    end

    dW = dW / 20;
    db = db / 20;
    
%     dw1 = dw1/20;
%     dw2 = dw2/20;
%     dw3 = dw3/20;
%     dw4 = dw4/20;
%     dw5 = dw5/20;
%     dw6 = dw6/20;
%     db1 = db1/20;
%     db2 = db2/20;
%     db3 = db3/20;

    dW(dW > threshold) = threshold;
    db(db > threshold) = threshold;

%     dw1(dw1 > threshold) = threshold;
%     dw2(dw2 > threshold) = threshold;
%     dw3(dw3 > threshold) = threshold;
%     dw4(dw4 > threshold) = threshold;
%     dw5(dw5 > threshold) = threshold;
%     dw6(dw6 > threshold) = threshold;
%     db1(db1 > threshold) = threshold;
%     db2(db2 > threshold) = threshold;
%     db3(db3 > threshold) = threshold;
    
    J = J/20
    
    W = W - learning_rate .* dW;
    b = b - learning_rate .* db;
    
%     w1 = w1 - learning_rate*dw1;
%     w2 = w2 - learning_rate*dw2;
%     w3 = w3 - learning_rate*dw3;
%     w4 = w4 - learning_rate*dw4;
%     w5 = w5 - learning_rate*dw5;
%     w6 = w6 - learning_rate*dw6;
%     b1 = b1 - learning_rate*db1;
%     b2 = b2 - learning_rate*db2;
%     b3 = b3 - learning_rate*db3;
    
    Z = zeros(size(X1));
    for i = 1 : size(W, 2)
        Z = Z + X1.*W(1, i) + X2.*W(2, i) + b(i);
        Z(Z < 0) = 0;
    end

%     Z1 = X1.*W(1, 1) + X2.*W(2, 1) + b(1);
%     Z2 = X1.*W(1, 2) + X2.*W(2, 2) + b(2);
%     Z3 = X1.*W(1, 3) + X2.*W(2, 3) + b(3);
% 
%     Z1(Z1 < 0) = 0;
%     Z2(Z2 < 0) = 0;
%     Z3(Z3 < 0) = 0;
% 
%     A1 = Z1;
%     A2 = Z2;
%     A3 = Z3;
% 
%     Y_cap = A1 + A2 + A3;
    
    surf_handle.ZData = Z;
    pause(0.01);
end