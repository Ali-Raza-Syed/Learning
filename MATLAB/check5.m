% two inputs with 2 first layer nodes with thresholding and ReLU activation

w1 = 2e-1;
w2 = 3e-1;
w3 = 2e-1;
w4 = 3e-1;
b1 = 1e-1;
b2 = 1e-1;
learning_rate = 1e-2;
x1 = -1:0.02:1;
x2 = -1:0.02:1;
N = 20;
threshold = 1;
points_range = [0, 1];
p_x1 = points_range(1) + (points_range(2)-points_range(1)).*rand(1, N);
p_x2 = points_range(1) + (points_range(2)-points_range(1)).*rand(1, N);
p_z = points_range(1) + (points_range(2)-points_range(1)).*rand(1, N);
dw1 = 0;
dw2 = 0;
dw3 = 0;
dw4 = 0;
db1 = 0;
db2 = 0;

[X1,X2] = meshgrid(x1, x2);

Z1 = X1.*w1 + X2.*w2 + b1;
Z2 = X1.*w3 + X2.*w4 + b2;

Z1(Z1 < 0) = 0;
Z2(Z2 < 0) = 0;

A1 = Z1;
A2 = Z2;

Y_cap = A1 + A2;

scatter_handle = scatter3(p_x1, p_x2, p_z, 'black');
hold on
surf_handle = surf(X1, X2, Y_cap);
colormap(gca(), lines(3));
surf_handle.FaceAlpha = 1;
axis([points_range(1), points_range(2), points_range(1), points_range(2), points_range(1), points_range(2)]);
while true
    dw1 = 0;
    dw2 = 0;
    dw3 = 0;
    dw4 = 0;
    db1 = 0;
    db2 = 0;
    J = 0;
    for i = 1:N
        z1 = p_x1(i)*w1 + p_x2(i)*w2 + b1;
        z2 = p_x1(i)*w3 + p_x2(i)*w4 + b2;
        
        z1(z1 < 0) = 0;
        z2(z2 < 0) = 0;
        
        a1 = z1;
        a2 = z2;
        
        y_cap = a1 + a2;
        
        update_flag_z1 = z1 > 0;
        update_flag_z2 = z2 > 0;
        
        J = J + (y_cap - p_z(i))^2;
        dw1 = dw1 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z1;
        dw2 = dw2 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z1;
        dw3 = dw3 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z2;
        dw4 = dw4 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z2;
        db1 = db1 + 2*(y_cap - p_z(i))*update_flag_z1;
        db2 = db2 + 2*(y_cap - p_z(i))*update_flag_z2;
        
    end
    
    dw1 = dw1/N;
    dw2 = dw2/N;
    dw3 = dw3/N;
    dw4 = dw4/N;
    db1 = db1/N;
    db2 = db2/N;
    
    dw1(dw1 > threshold) = threshold;
    dw2(dw2 > threshold) = threshold;
    dw3(dw3 > threshold) = threshold;
    dw4(dw4 > threshold) = threshold;
    db1(db1 > threshold) = threshold;
    db2(db2 > threshold) = threshold;
    
    J = J/N
    w1 = w1 - learning_rate*dw1;
    w2 = w2 - learning_rate*dw2;
    w3 = w3 - learning_rate*dw3;
    w4 = w4 - learning_rate*dw4;

    b1 = b1 - learning_rate*db1;
    b2 = b2 - learning_rate*db2;
    
    Z1 = X1.*w1 + X2.*w2 + b1;
    Z2 = X1.*w3 + X2.*w4 + b2;

    Z1(Z1 < 0) = 0;
    Z2(Z2 < 0) = 0;

    A1 = Z1;
    A2 = Z2;
    
    Y_cap = A1 + A2;
    
    
    surf_handle.ZData = Y_cap;
    pause(0.1);
end