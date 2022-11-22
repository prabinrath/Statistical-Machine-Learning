[x,y] = meshgrid(-8:.1:8);
mu = [0;0];
sigma = eye(2)*5;
z = zeros(size(x));
for i=1:size(z,1)
    for j=1:size(z,2)
        q = [x(i,j);y(i,j)];
        z(i,j) = gauss(q,mu,sigma);
    end
end
mesh(x,y,z);