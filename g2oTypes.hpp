#include <Eigen/Dense>

#include <sophus/se3.hpp>

#include "g2o/core/sparse_optimizer.h"
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>

template <typename T = double>
Eigen::Matrix<T, 3, 3> NormalizeRotation(const Eigen::Matrix<T, 3, 3> &R)
{
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return NormalizeRotation(res);
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return NormalizeRotation(res);
    }
}

struct CameraModel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    float f = 0.008; // 焦距
    float pix_x_meter = 2e-5;
    float pix_y_meter = 2e-5;
    float fx = f / pix_x_meter;
    float fy = f / pix_y_meter;
    int img_width = 1920;
    int img_height = 1080;
    int cx = img_width / 2;
    int cy = img_height / 2;
    Eigen::Matrix3f K;
    //K << fx, cx, 0,
    //	0, fy, cy,
    //	0, 0, 1;
    Eigen::Matrix3f K_inv; // = K.inverse();
    CameraModel(){
        K << fx, cx, 0,
        0, fy, cy,
        0, 0, 1;
        K_inv = K.inverse();
    }
};

// point 类型
class PointXYZ {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointXYZ(){};
    PointXYZ(double _x, double _y, double _z){
        xyz = Eigen::Vector3d(_x, _y, _z);
    }
    PointXYZ(const Eigen::Vector3d& p){
        xyz = p;
    }
    // 	需要定义Update函数
    void Update(const double *up){
        Eigen::Vector3d delta(up[0], up[1], up[2]);
        xyz += delta;
    }
    Eigen::Vector3d xyz;
};
class VertexPointXYZ : public g2o::BaseVertex<3, PointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPointXYZ(){};
    VertexPointXYZ(double _x, double _y, double _z){
        PointXYZ xyz(_x, _y, _z);
        setEstimate(xyz);
    }
    VertexPointXYZ(const Eigen::Vector3d& p){
        PointXYZ xyz = p;
        setEstimate(xyz);
    }

    // 需要重新定义的纯虚函数
    virtual bool read(std::istream &is){return false;}
    virtual bool write(std::ostream &os) const{return false;}

    // 重置函数,设定被优化变量的原始值
    virtual void setToOriginImpl()
    {
    }

    virtual void oplusImpl(const double *update_)
    {
        // https://github.com/RainerKuemmerle/g2o/blob/master/doc/README_IF_IT_WAS_WORKING_AND_IT_DOES_NOT.txt
        // 官方讲解cache
        // 需要在oplusImpl与setEstimate函数中添加
        _estimate.Update(update_);
        updateCache();
    }

};


// pose 类型
class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Pose(){};
    Pose(Sophus::SE3d _T):T(_T){}
    void Update(const double *up){
        Eigen::Vector3d r(up[0], up[1], up[2]);
        Eigen::Vector3d t(up[3], up[4], up[5]);
        Eigen::Matrix3d delta_r = Eigen::AngleAxisd(r.norm(), r.normalized()).toRotationMatrix();
        // Eigen::Matrix3d delta_r = ExpSO3(r[0], r[1], r[2]);

        // 这样赋值是错误的，rotationMatrix()函数是const修饰，返回临时变量
        // T.rotationMatrix() = delta_r * T.rotationMatrix();
        T.setRotationMatrix(delta_r * T.rotationMatrix());
        T.translation() += t;
    }
    Sophus::SE3d T;
};
class VertexPose : public g2o::BaseVertex<6, Pose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){};
    VertexPose(Sophus::SE3d _T){
        Pose T(_T);
        setEstimate(_T);
    }

    virtual bool read(std::istream &is){return false;}
    virtual bool write(std::ostream &os)const {return false;}

    // 重置函数,设定被优化变量的原始值
    virtual void setToOriginImpl()
    {
    }

    virtual void oplusImpl(const double *update_)
    {
        _estimate.Update(update_);
        updateCache();
    }
};

// point->pose类型
class EdgeMono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexPointXYZ>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMono(){};

    EdgeMono(const Eigen::Vector2d _p, CameraModel* _cam) : cam(_cam)
    {
        // 像素值为int类型
        setMeasurement(_p.cast<int>().cast<double>());
    }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    // 计算重投影误差
    void computeError()
    {
        const VertexPose* Tcw = static_cast<VertexPose*>(_vertices[0]);
        const VertexPointXYZ* pw = static_cast<VertexPointXYZ*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        
        pc1 = Tcw->estimate().T * pw->estimate().xyz;
        pc2 = cam->K.cast<double>() * pc1;
        Eigen::Vector2d pix = (pc2 / pc2[2]).head(2);
        _error = pix - obs;
        // std::cout<<"e: "<<_error.transpose()<<std::endl;       
    }

    virtual void linearizeOplus(){
        const VertexPose* Tcw = static_cast<VertexPose*>(_vertices[0]);
        const VertexPointXYZ* _pw = static_cast<VertexPointXYZ*>(_vertices[1]);

        const Eigen::Matrix3d Rbw = Tcw->estimate().T.rotationMatrix();
        const Eigen::Vector3d tbw = Tcw->estimate().T.translation();
        const Eigen::Vector3d pw = _pw->estimate().xyz;

        // residual [r1, r2] about pc2[px, py, pz]
        Eigen::Matrix<double, 2, 3> J_r_pc2;
        J_r_pc2 << 1/pc2[2], 0, -pc2[0]/(pc2[2]*pc2[2]),
                    0, 1/pc2[2], -pc2[1]/(pc2[2]*pc2[2]);
        // pc2 about pc1
        // const double cx = cam->cx, cy = cam->cy, fx = cam->fx, fy = cam->fy;
        Eigen::Matrix3d J_pc2_pc1 = cam->K.cast<double>();
        // J_pc2_pc1<< cx, 0, fx, \
                    0, cy, fy, \
                    0, 0, 1;
        // pc1 about pw
        Eigen::Matrix3d J_pc1_pw = Rbw;
        _jacobianOplusXj = J_r_pc2 * J_pc2_pc1 * J_pc1_pw;

        // Jacobian [r1, r2] about [R, t]
        // pc1 w.r.t. Rbw
        Eigen::Matrix3d J_pc1_Rbw = Sophus::SO3d::hat(-Rbw * pw);
        // pc1 w.r.t tbw
        Eigen::Matrix3d J_pc1_tbw = Eigen::Matrix3d::Identity();
        // pc1 w.r.t. Tbw
        Eigen::Matrix<double, 3, 6> J_pc1_Tbw;
        J_pc1_Tbw.block<3, 3>(0, 0) = J_pc1_Rbw;
        J_pc1_Tbw.block<3, 3>(0, 3) = J_pc1_tbw;
        _jacobianOplusXi = J_r_pc2 * J_pc2_pc1 * J_pc1_Tbw;
    }
    CameraModel* cam{nullptr};

    Eigen::Vector3d pc1, pc2;
    Eigen::Vector3d pix;
};