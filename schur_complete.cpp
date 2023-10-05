#include <iostream>
#include <vector>

#include <Eigen/Dense>

int main(){
    // residual w.r.t pose[i]
    std::vector<Eigen::Matrix<double, 2, 6>> poseJs(9, Eigen::Matrix<double, 2, 6>::Zero());
    poseJs[0] << -248.767, 360.012, 725.936, 214.451, 514.683, -181.757,
                 -137.973, -36.6332, 273.126, 0, 214.451, 28.7634;
    poseJs[1] << -774.743, 17.2344, -642.311, 137.521, 330.051, -157.02,
                 -335.102, -97.9796, -182.764, 0, 137.521, -73.7248;
    poseJs[2] << -1131.02, -279.955, -1367.48, 96.4626, 231.51, -127.178,
                 -594.297, -380.331, -423.116, 0, 96.4626, -86.7084;
    poseJs[3] << -98.1715, 800.778, 1485.34, 229.295, 550.308, -281.526,
                 -148.956, -164.066, 562.833, 0, 229.295, 66.8396;
    poseJs[4] << -893.866, 80.7091, -478.566, 131.317, 315.161, -192.123,
                 -368.99, -62.185, -104.615, 0, 131.317, -78.0572;
    poseJs[5] << -1297.08, -340.017, -1352.3, 88.393, 212.143, -138.124,
                 -670.342, -396.432, -401.967, 0, 88.393, -87.1759;
    poseJs[6] << 327.796, 2205.51, 2828, 266.929, 640.63, -530.556,
                 -193.44, -590.425, 1078.38, 0, 266.929, 146.147;
    poseJs[7] << -1032.69, 254.828, -222.822, 131.668, 316.004, -248.835,
                 -396.183, 7.40965, 11.3802, 0, 131.668, -85.7289;
    poseJs[8] << -1493.48, -384.085, -1297.67, 83.9361, 201.447, -156.226,
                 -749.372, -391.682, -364.181, 0, 83.9361, -90.2744;

    // residual w.r.t point[i]
    std::vector<Eigen::Matrix<double, 2, 3>> pointJs(9, Eigen::Matrix<double, 2, 3>::Zero());
    pointJs[0] << 170.303, 552.652, -97.4487,
                 -46.9273, 209.485, 27.0287;
    pointJs[1] << 122.84, 356.802, -100.515,
                  0.951991, 139.811, -69.2787;
    pointJs[2] << 91.9033, 251.287, -86.5159,
                  12.4549, 100.147, -81.4792;
    pointJs[3] << 211.458, 596.083, -186.125,
                  -62.3298, 221.842, 62.8087;
    pointJs[4] << 131.508, 343.211, -135.624,
                  3.48857, 133.958, -73.3497;
    pointJs[5] << 91.4857, 231.547, -99.5623,
                  14.0137, 92.2275, -81.9186;
    pointJs[6] << 314.48, 705.964, -407.264,
                  -95.5776, 254.194, 137.333;
    pointJs[7] << 150.788, 347.466, -188.795,
                  6.01161, 134.759, -80.5588;
    pointJs[8] << 95.3156, 221.361, -118.096,
                  15.8313, 88.0224, -84.8302;

    // 1pose + 9point
    Eigen::Matrix<double, 18, 33> J;
    J.setZero();
    for(int i=0; i<poseJs.size(); ++i){
        J.block<2, 6>(i*2, 0) = poseJs[i];
        J.block<2, 3>(i*2, 6+i*3) = pointJs[i];
    }
    std::cout<<"J(18, 24):\n"<<J.cast<int>().block<18, 24>(0, 0)<<std::endl;
    std::cout<<"J(18, 25)"<<J.cast<int>().block<18, 9>(0, 24)<<std::endl;

    Eigen::Matrix<double, 18, 1> e;
    e << -44.0037, -9.22352, -40.9825, 15.4527, -30.5357, 31.8888, -30.9359, -9.68274, -25.8493, 17.6775, -21.4829, 
         33.6115, -7.14896, 1.72825, -15.7248, 18.738, -13.3746, 35.5083;
    
    // H * Δx = -J.T * r
    // H只是半正定，直接求逆的结果是错误的，这里为了使其正定，在对角线添加元素
    // 查Eigen病态矩阵判断及求逆 ： https://blog.csdn.net/tuntunmmd/article/details/104898610
    // 矩阵病态问题和条件数： https://blog.csdn.net/LVXIAO2897/article/details/102443382
    Eigen::Matrix<double, 33, 33> H = J.transpose() * J;
    std::cout<<"H.cond = "<< H.colPivHouseholderQr().absDeterminant() / H.diagonal().prod() <<std::endl;
	Eigen::Matrix<double, 33, 33> lambdaMatrix = Eigen::Matrix<double, 33, 33>::Identity();
    H += lambdaMatrix;
    std::cout<<"(H+I).cond = "<< H.colPivHouseholderQr().absDeterminant() / H.diagonal().prod() <<std::endl;

    Eigen::Matrix<double, 33, 1> b = -J.transpose() * e;
    Eigen::Matrix<double, 33, 1> delta = H.inverse() * b;
    Eigen::Matrix<int, 33, 1> delta_i = delta.cast<int>();
    std::cout<<"delta:\n"<<delta.head(6).transpose()<<std::endl<<delta.transpose().tail(27)<<std::endl;
    std::cout<<"I:\n"<<(H.inverse()*H).cast<int>()<<std::endl;
    
    /***
    *     T   p...
    * T   A   B
    * p   C   D
    * ...
    ***/
    // 对B进行边缘化，左乘形成上三角矩阵
    // | I          0 |   | A  B |   | A  B |
    // | -C*A.inv   I | * | C  D | = | 0  ΔA| ==> ΔA = -C*A.inv*B + D
	// 对C进行边缘化，右乘形成下三角矩阵
    // | A  B |   | I  -A.inv*B |   | A  0 |
    // | C  D | * | 0       I   | = | C  ΔA| = H' ==> 用来求pose
    // 可以得到:
    // | I        0 |   | A  B |   | I  -A.inv*B |   | A  0 |
    // |-C*A.inv  I | * | C  D | * | 0      I    | = | 0  ΔA| = H'
	
	// 自己可以构建舒尔补，左乘形成下三角矩阵，先求解pose增量，再求解point增量
	// | I  -B*D.inv |   | A  B |   | A-B*D.inv*C  0 |
	// | 0     I     | * | C  D | = |    C         D |
	// H -= lambdaMatrix;
    Eigen::Matrix<double, 6, 6> A = H.block<6, 6>(0, 0);
    Eigen::Matrix<double, 6, 27> B = H.block<6, 27>(0, 6);
    Eigen::Matrix<double, 27, 6> C = H.block<27, 6>(6, 0);
    // std::cout<<"B - C.T:\n"<<B-C.transpose()<<std::endl;
    Eigen::Matrix<double, 27, 27> D = H.block<27, 27>(6, 6); 
    
    Eigen::Matrix<double, 6, 27> E = -B * D.inverse();
    Eigen::Matrix<double, 33, 33> leftMatrix;
    leftMatrix.setIdentity();
    leftMatrix.block<6, 27>(0, 6) = E;

	// 求pose增量
	Eigen::Matrix<double, 6, 6> deltaA = A + E * C;
	Eigen::Matrix<double, 6, 1> deltaX_pose = deltaA.inverse() * (leftMatrix * b).head(6);
	std::cout<<"deltaPose diff:\n"<<(deltaX_pose - delta.head(6)).transpose().cast<int>()<<std::endl;
    // 求point增量
    // H * Δx = b ==> C*deltaX_pose + D*deltaX_point = b
	// D*deltaX_point = b - C*deltaX_pose
	// deltaX_point = D.inv * (b - C*deltaX_pose)
	// D是对角块矩阵，可以分块求逆
	Eigen::Matrix<double, 27, 27> D_inv;
	D_inv.setZero();
	for(int i=0; i<D_inv.rows()/3; ++i){
		D_inv.block<3, 3>(i*3, i*3) = D.block<3, 3>(i*3, i*3).inverse();
	}
	Eigen::Matrix<double, 27, 1> deltaX_point = D_inv * (b.tail(27) - C*deltaX_pose);
    std::cout<<"deltaPointX diff:\n"<<(deltaX_point - delta.tail(27)).transpose().cast<int>()<<std::endl;

    return 0;

}