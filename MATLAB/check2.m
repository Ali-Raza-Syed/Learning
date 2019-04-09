% two inputs with 3 first layer nodes with thresholding and ReLU activation
seed = 1;
rng(seed);
w1 = 2e-1;
w2 = 3e-1;
b1 = 1e-1;
w3 = -3e-1;
w4 = -2e-1;
b2 = 10e-1;
w5 = -5e-1;
w6 = 1e-1;
b3 = 10e-1;
learning_rate = 1e-2;
x1 = -1:0.02:1;
x2 = -1:0.02:1;

threshold = 1;
rng(seed);
p_x1 = 0 + (1-0).*rand(1, 20);
rng(seed);
p_x2 = 0 + (1-0).*rand(1, 20);
rng(seed);
p_z = 0 + (1-0).*rand(1, 20);
dw1 = 0;
dw2 = 0;
dw3 = 0;
dw4 = 0;
dw5 = 0;
dw6 = 0;
db1 = 0;
db2 = 0;
db3 = 0;

[X1,X2] = meshgrid(x1, x2);

Z1 = X1.*w1 + X2.*w2 + b1;
Z2 = X1.*w3 + X2.*w4 + b2;
Z3 = X1.*w5 + X2.*w6 + b3;

Z1(Z1 < 0) = 0;
Z2(Z2 < 0) = 0;
Z3(Z3 < 0) = 0;

A1 = Z1;
A2 = Z2;
A3 = Z3;

Y_cap = A1 + A2 + A3;
[dfdx,dfdy] = gradient(Y_cap);

scatter_handle = scatter3(p_x1, p_x2, p_z, 'black');
hold on
surf_handle = surf(X1, X2, Y_cap);
colormap(gca(), lines(8));
surf_handle.FaceAlpha = 0.5;
axis([-1, 1, -1, 1, -1, 1]);
while true
    dw1 = 0;
    dw2 = 0;
    dw3 = 0;
    dw4 = 0;
    dw5 = 0;
    dw6 = 0;
    db1 = 0;
    db2 = 0;
    db3 = 0;
    J = 0;
    for i = 1:20
        z1 = p_x1(i)*w1 + p_x2(i)*w2 + b1;
        z2 = p_x1(i)*w3 + p_x2(i)*w4 + b2;
        z3 = p_x1(i)*w5 + p_x2(i)*w6 + b3;
        
        z1(z1 < 0) = 0;
        z2(z2 < 0) = 0;
        z3(z3 < 0) = 0;
        
        a1 = z1;
        a2 = z2;
        a3 = z3;
        
        y_cap = a1 + a2 + a3;
        
        update_flag_z1 = z1 > 0;
        update_flag_z2 = z2 > 0;
        update_flag_z3 = z3 > 0;
        
        J = J + (y_cap - p_z(i))^2;
        dw1 = dw1 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z1;
        dw2 = dw2 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z1;
        dw3 = dw3 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z2;
        dw4 = dw4 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z2;
        dw5 = dw5 + 2*(y_cap - p_z(i))*p_x1(i) * update_flag_z3;
        dw6 = dw6 + 2*(y_cap - p_z(i))*p_x2(i) * update_flag_z3;
        db1 = db1 + 2*(y_cap - p_z(i))*update_flag_z1;
        db2 = db2 + 2*(y_cap - p_z(i))*update_flag_z2;
        db3 = db3 + 2*(y_cap - p_z(i))*update_flag_z3;
        
    end
    
    dw1 = dw1/20;
    dw2 = dw2/20;
    dw3 = dw3/20;
    dw4 = dw4/20;
    dw5 = dw5/20;
    dw6 = dw6/20;
    db1 = db1/20;
    db2 = db2/20;
    db3 = db3/20;
    
    dw1(dw1 > threshold) = threshold;
    dw2(dw2 > threshold) = threshold;
    dw3(dw3 > threshold) = threshold;
    dw4(dw4 > threshold) = threshold;
    dw5(dw5 > threshold) = threshold;
    dw6(dw6 > threshold) = threshold;
    db1(db1 > threshold) = threshold;
    db2(db2 > threshold) = threshold;
    db3(db3 > threshold) = threshold;
    
    J = J/20
    w1 = w1 - learning_rate*dw1;
    w2 = w2 - learning_rate*dw2;
    w3 = w3 - learning_rate*dw3;
    w4 = w4 - learning_rate*dw4;
    w5 = w5 - learning_rate*dw5;
    w6 = w6 - learning_rate*dw6;
    b1 = b1 - learning_rate*db1;
    b2 = b2 - learning_rate*db2;
    b3 = b3 - learning_rate*db3;
    
    Z1 = X1.*w1 + X2.*w2 + b1;
    Z2 = X1.*w3 + X2.*w4 + b2;
    Z3 = X1.*w5 + X2.*w6 + b3;

    Z1(Z1 < 0) = 0;
    Z2(Z2 < 0) = 0;
    Z3(Z3 < 0) = 0;

    A1 = Z1;
    A2 = Z2;
    A3 = Z3;

    Y_cap = A1 + A2 + A3;
    
    
    surf_handle.ZData = Y_cap;
    pause(0.01);
end