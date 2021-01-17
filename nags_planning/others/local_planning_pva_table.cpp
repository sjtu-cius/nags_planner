#include <ewok/ed_nor_ring_buffer.h>

#include <ros/ros.h>

#include <tf/transform_datatypes.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int16.h>
#include <visualization_msgs/Marker.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cstdint>

using namespace message_filters;

#define GRAVATY 9.8

// global declaration
ros::Time _data_input_time;
ros::Time _algorithm_time;

bool initialized = false;
const double resolution = 0.2;

static const int POW = 6;
static const int N = (1 << POW);

const float cal_duration = 0.05;

ewok::EuclideanDistanceNormalRingBuffer<POW> rrb(resolution, 1.0);

ros::Publisher current_marker_pub;
ros::Publisher cloud2_pub, center_pub;

ros::Publisher traj_point_pub; // add on 9 Mar
ros::Publisher vis_cloud_env_pub, vis_cloud_dis_field_pub;
ros::Publisher p_goal_pose_pub;

ros::Publisher control_p_idx_pub;

double x_centre, y_centre, z_centre;

bool objects_updated = false;
bool odom_initilized = false;
bool imu_initilized = false;
bool state_locked = false;
bool state_updating = false;
bool in_safety_mode = false;
bool safety_mode_recover = false;
bool uav_pause = true;

/****** Parameters for path planning ******/
const int ANGLE_H_NUM = 13;
const int ANGLE_V_NUM = 1;
Eigen::VectorXd Fov_half(2); //Fov parameters
Eigen::VectorXd Angle_h(ANGLE_H_NUM);  // initiate later in the main function
Eigen::VectorXd Angle_v(ANGLE_V_NUM); // initiate later in the main function
Eigen::MatrixXd F_cost;

Eigen::Vector3d p_goal;
Eigen::Vector3d p0;
Eigen::Vector3d v0;
Eigen::Vector3d a0;
double yaw0 = 0.0;
double theta_h_last = 0.0;
double theta_v_last = 0.0;

Eigen::Vector3d p_store;
double yaw_store = 0.0;
bool test_angle = false;
double th_h = 0.0;
double ph_h = 0.0;

// When flying outdoors, p_goal_radius and flight_altitude can be larger.
double p_goal_radius = 50.0;
double flight_altitude = 1.5;
double rc_theta = 0.0;

// Only for debugging
// 2 maps if set cloud_1 as a global variable
// pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGB>());

bool odom_init = false;
Eigen::Vector3d odom_pose;
Eigen::Quaternionf odom_q;

// Only for checking acceleration
ros::Publisher max_accel_pub;

// PVA table for feasible total planning time T
typedef struct pva_table pva_table;
struct pva_table {
    int dim_num;
    int *dim_size;
    uint64_t *dim_interval;
    uint64_t table_size;
    double *table;
    double rez;
    double *pva_limit;

    void construct_pva_table(int dim1_size, int dim2_size, int dim3_size, int dim4_size, double resolution) {
        this->dim_num = 4;
        this->dim_size = (int*)malloc(sizeof(int)*this->dim_num);
        this->dim_interval = (uint64_t*)malloc(sizeof(uint64_t)*this->dim_num);

        this->dim_size[0] = dim1_size;
        this->dim_size[1] = dim2_size;
        this->dim_size[2] = dim3_size;
        this->dim_size[3] = dim4_size;

        this->dim_interval[3] = 1;
        this->dim_interval[2] = dim_interval[3] * dim4_size;
        this->dim_interval[1] = dim_interval[2] * dim3_size;
        this->dim_interval[0] = dim_interval[1] * dim2_size;

        this->table_size = this->dim_interval[0] * dim1_size;
        this->table = (double*)malloc(sizeof(double)*this->table_size);

        this->rez = resolution;

        this->pva_limit = (double*)malloc(sizeof(double)*3);
        this->pva_limit[0] = this->rez*double(dim1_size/2);
        this->pva_limit[1] = this->rez*double(dim2_size/2);
        this->pva_limit[2] = this->rez*double(dim4_size/2);
    }

    void compute_idx_from_pva(double dlt_p, double v0, double vf, double a0,
                              int &idx1, int &idx2, int &idx3, int &idx4) {
        idx1 = round(dlt_p/this->rez) + this->dim_size[0]/2;
        idx2 = round(v0/this->rez) + this->dim_size[1]/2;
        idx3 = round(vf/this->rez) + this->dim_size[2]/2;
        idx4 = round(a0/this->rez) + this->dim_size[3]/2;
    }

    double query_pva_table(double dlt_p, double v0, double vf, double a0) {
        if (fabs(dlt_p) > this->pva_limit[0]) static_assert("Invalid input!", "");
        if (fabs(v0) > this->pva_limit[1]) static_assert("Invalid input!", "");
        if (fabs(vf) > this->pva_limit[1]) static_assert("Invalid input!", "");
        if (fabs(a0) > this->pva_limit[2]) static_assert("Invalid input!", "");

        int idx1, idx2, idx3, idx4;
        this->compute_idx_from_pva(dlt_p, v0, vf, a0, idx1, idx2, idx3, idx4);

        uint64_t idx = idx1*this->dim_interval[0] + idx2*this->dim_interval[1] +
                       idx3*this->dim_interval[2] + idx4*this->dim_interval[3];

        // std::cout << "idx: " << idx << std::endl;

        return this->table[idx];
    }

    void pva_table2csv(const std::string &str) {
        std::ofstream outfile;
        outfile.open(str, std::ios::out);

        for (int i = 0; i < 4; ++i) outfile << std::to_string(this->dim_size[i]) << ',';
        outfile << std::to_string(this->rez) << std::endl;

        for (uint64_t i = 0; i < this->table_size-1; ++i) outfile << std::to_string(this->table[i]) << ',';
        outfile << std::to_string(this->table[table_size-1]);

        outfile.close();
    }

    void csv2pva_table(const std::string &str) {
        int tmp_dim_size[4];
        double tmp_rez;

        std::ifstream infile(str, std::ios::in);
        std::string tmp_str;

        for (int i = 0; i < 4; ++i) {
            getline(infile, tmp_str, ',');
            tmp_dim_size[i] = std::stoi(tmp_str);
        }

        getline(infile, tmp_str);
        tmp_rez = std::stod(tmp_str);

        this->construct_pva_table(tmp_dim_size[0], tmp_dim_size[1],
                                  tmp_dim_size[2], tmp_dim_size[3], tmp_rez);

        for (uint64_t i = 0; i < this->table_size; ++i) {
            getline(infile, tmp_str, ',');
            this->table[i] = std::stod(tmp_str);
        }
    }

    void free_pva_table() {
        free(this->pva_limit);
        free(this->table);
        free(this->dim_interval);
        free(this->dim_size);
    }
};

// Test
pva_table *table = (pva_table*)malloc(sizeof(pva_table));
bool table_initialized = false;

struct  Path_Planning_Parameters
{
    double d_ref = 3.0;
    double k1_xy = 2; //% Goal directed coefficient
    double k1_z = 2; //% Goal directed coefficient
    double k2_xy = 4; //% Rotation coefficient
    double k2_z = 4; //% Rotation coefficient
    double k3 = M_PI*0.1; //% FOV coefficient
    double kk_h = 1; //% FOV horisontal cost coefficient
    double kk_v = 1; //% FOV vertical cost coefficient
    double v_max_ori = 2; //% m/s, just reference
    double v_scale_min = 0.1;
    double delt_t = 0.05; //%time interval between two control points
    int max_plan_num = ANGLE_H_NUM * ANGLE_V_NUM;  // Previously it was 100, as 18*7 = 126 > 100
}pp;

/**************************************************/

// // this callback use input cloud to update ring buffer, and update odometry of UAV
// void odomCloudCallback(const nav_msgs::OdometryConstPtr& odom, const sensor_msgs::PointCloud2ConstPtr& cloud)
// {
//     ROS_INFO("Received Point Cloud!");
//     ROS_WARN("Callback!!!!!!!!!!");
//     _data_input_time = ros::Time::now();

//     tf::Quaternion q1(odom->pose.pose.orientation.x, odom->pose.pose.orientation.y,
//                       odom->pose.pose.orientation.z, odom->pose.pose.orientation.w);
//     tf::Matrix3x3 m(q1);
//     double roll, pitch, yaw;
//     m.getRPY(roll, pitch, yaw);

//     // update ring buffer
//     // tranform from optical frame to uav frame
//     Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
//     t_c_b(0, 2) = 1;
//     t_c_b(1, 0) = -1;
//     t_c_b(2, 1) = -1;
//     t_c_b(3, 3) = 1;

//     // transform from uav to world
//     // get orientation and translation
//     Eigen::Quaternionf q;
//     q.w() = odom->pose.pose.orientation.w;
//     q.x() = odom->pose.pose.orientation.x;
//     q.y() = odom->pose.pose.orientation.y;
//     q.z() = odom->pose.pose.orientation.z;

//     // create transform matrix
//     Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
//     transform.block(0, 0, 3, 3) = Eigen::Matrix3f(q);
//     transform(0, 3) = odom->pose.pose.position.x;
//     transform(1, 3) = odom->pose.pose.position.y;
//     transform(2, 3) = odom->pose.pose.position.z;
//     // std::cout << transform.matrix() << "\n\n";

//     // convert cloud to pcl form
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>());
//     pcl::fromROSMsg(*cloud, *cloud_in);
//     // transform to world frame
//     //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
//     //pcl::transformPointCloud(*cloud_in, *cloud_2, transform);
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGB>());
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
//     pcl::transformPointCloud(*cloud_in, *cloud_1, t_c_b);
//     pcl::transformPointCloud(*cloud_1, *cloud_2, transform);

//     // down-sample for all
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
//     pcl::VoxelGrid<pcl::PointXYZRGB> sor;
//     sor.setInputCloud(cloud_2);
//     float res = 0.1f;
//     sor.setLeafSize(res, res, res);
//     sor.filter(*cloud_filtered);

//     // compute ewol pointcloud and origin
//     Eigen::Vector3f origin = (transform * Eigen::Vector4f(0, 0, 0, 1)).head<3>(); // odom position (x,y,z)
//     ewok::EuclideanDistanceNormalRingBuffer<POW>::PointCloud cloud_ew;
//     std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > points =
//             cloud_filtered->points; //  cloud_2->points;

//     x_centre = odom->pose.pose.position.x;
//     y_centre = odom->pose.pose.position.y;
//     z_centre = odom->pose.pose.position.z;

//     for(int i = 0; i < points.size(); ++i)
//     {
//         double x_diff = fabs(x_centre - points.at(i).x);
//         double y_diff = fabs(y_centre - points.at(i).y);
//         double z_diff = fabs(z_centre - points.at(i).z);
//         double noise_threshold = x_diff*x_diff + y_diff*y_diff + z_diff*z_diff;

//         if (noise_threshold > 0.2)
//             cloud_ew.push_back(Eigen::Vector4f(points.at(i).x, points.at(i).y, points.at(i).z, 0));
//     }

//     // initialize the ringbuffer map
//     if(!initialized)
//     {
//         Eigen::Vector3i idx;
//         rrb.getIdx(origin, idx);
//         rrb.setOffset(idx);
//         initialized = true;
//     }
//     else
//     {
//         // move buffer when its center is not the same as UAV
//         while(true)
//         {
//             Eigen::Vector3i origin_idx, offset, diff;
//             rrb.getIdx(origin, origin_idx);
//             offset = rrb.getVolumeCenter();
//             //std::cout << "origin :" << origin_idx << " center:" << offset << std::endl;
//             diff = origin_idx - offset;
//             if(diff.array().any())
//                 rrb.moveVolume(diff.head<3>());
//             else
//                 break;
//         }
//     }

//     // insert point cloud to ringbuffer
//     rrb.insertPointCloud(cloud_ew, origin);

//     rrb.updateDistance();

//     double elp = ros::Time::now().toSec() - _data_input_time.toSec();
//     std::cout << "Map update time = " << elp << " s" << std::endl;
// }

void odomPclCallback(const nav_msgs::OdometryConstPtr& odom)
{
    // ROS_WARN("Odom!!!!!!!!!!");

    odom_q.w() = odom->pose.pose.orientation.w;
    odom_q.x() = odom->pose.pose.orientation.x;
    odom_q.y() = odom->pose.pose.orientation.y;
    odom_q.z() = odom->pose.pose.orientation.z;

    odom_pose(0) = odom->pose.pose.position.x;
    odom_pose(1) = odom->pose.pose.position.y;
    odom_pose(2) = odom->pose.pose.position.z;

    if (!odom_init) odom_init = true;
}

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    if (!odom_init) return;

    // ROS_WARN("Cloud!!!!!!!!!!");
    ROS_INFO("Received Point Cloud!");
    _data_input_time = ros::Time::now();

    tf::Quaternion q1(odom_q.x(), odom_q.y(), odom_q.z(), odom_q.w());
    tf::Matrix3x3 m(q1);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    // update ring buffer
    // tranform from optical frame to uav frame
    Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
    t_c_b(0, 2) = 1;
    t_c_b(1, 0) = -1;
    t_c_b(2, 1) = -1;
    t_c_b(3, 3) = 1;

    // transform from uav to world
    // get orientation and translation
    Eigen::Quaternionf q;
    q = odom_q;

    // create transform matrix
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = Eigen::Matrix3f(q);
    transform(0, 3) = odom_pose(0);
    transform(1, 3) = odom_pose(1);
    transform(2, 3) = odom_pose(2);
    // std::cout << transform.matrix() << "\n\n";

    // convert cloud to pcl form
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*cloud, *cloud_in);
    // transform to world frame
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
    //pcl::transformPointCloud(*cloud_in, *cloud_2, transform);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud_in, *cloud_1, t_c_b);
    pcl::transformPointCloud(*cloud_1, *cloud_2, transform);

    // down-sample for all
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_2);
    float res = 0.1f;
    sor.setLeafSize(res, res, res);
    sor.filter(*cloud_filtered);

    // compute ewol pointcloud and origin
    Eigen::Vector3f origin = (transform * Eigen::Vector4f(0, 0, 0, 1)).head<3>(); // odom position (x,y,z)
    ewok::EuclideanDistanceNormalRingBuffer<POW>::PointCloud cloud_ew;
    std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > points =
            cloud_filtered->points; //  cloud_2->points;

    x_centre = odom_pose(0);
    y_centre = odom_pose(1);
    z_centre = odom_pose(2);

    for(int i = 0; i < points.size(); ++i)
    {
        double x_diff = fabs(x_centre - points.at(i).x);
        double y_diff = fabs(y_centre - points.at(i).y);
        double z_diff = fabs(z_centre - points.at(i).z);
        double noise_threshold = x_diff*x_diff + y_diff*y_diff + z_diff*z_diff;

        if (noise_threshold > 0.2)
            cloud_ew.push_back(Eigen::Vector4f(points.at(i).x, points.at(i).y, points.at(i).z, 0));
    }

    // initialize the ringbuffer map
    if(!initialized)
    {
        Eigen::Vector3i idx;
        rrb.getIdx(origin, idx);
        rrb.setOffset(idx);
        initialized = true;
    }
    else
    {
        // move buffer when its center is not the same as UAV
        while(true)
        {
            Eigen::Vector3i origin_idx, offset, diff;
            rrb.getIdx(origin, origin_idx);
            offset = rrb.getVolumeCenter();
            //std::cout << "origin :" << origin_idx << " center:" << offset << std::endl;
            diff = origin_idx - offset;
            if(diff.array().any())
                rrb.moveVolume(diff.head<3>());
            else
                break;
        }
    }

    // insert point cloud to ringbuffer
    rrb.insertPointCloud(cloud_ew, origin);

    rrb.updateDistance();

    double elp = ros::Time::now().toSec() - _data_input_time.toSec();
    std::cout << "Map update time = " << elp << " s" << std::endl;
}

void timerCallback(const ros::TimerEvent& e)
{
    if(!initialized) return;

    /*Obstacle cloud*/
    pcl::PointCloud<pcl::PointXYZ> cloud;
    Eigen::Vector3d center;
    rrb.getBufferAsCloud(cloud, center);

    // convert to ROS message and publish
    sensor_msgs::PointCloud2 cloud2;
    pcl::toROSMsg(cloud, cloud2);

    // message publish should have the same time stamp
    cloud2.header.stamp = ros::Time::now();
    cloud2.header.frame_id = "world";
    cloud2_pub.publish(cloud2);

    //publish center
    geometry_msgs::PointStamped center_p;
    center_p.header = cloud2.header;
    center_p.point.x = center(0);
    center_p.point.y = center(1);
    center_p.point.z = center(2);
    center_pub.publish(center_p);
}

// void motion_primitives(Eigen::Vector3d p0, Eigen::Vector3d v0, Eigen::Vector3d a0, double yaw0, double theta_h,
//                        double theta_v, Eigen::Vector3d goal, double d, double v_max, double delt_t,
//                        Eigen::MatrixXd &p, Eigen::MatrixXd &v, Eigen::MatrixXd &a, Eigen::VectorXd &t)
// {
//     double delt_x = d*cos(theta_v)*cos(theta_h+yaw0);
//     double delt_y = d*cos(theta_v)*sin(theta_h+yaw0);
//     double delt_z = d*sin(theta_v);

//     Eigen::Vector3d pf;
//     pf << p0(0)+delt_x, p0(1)+delt_y, p0(2)+delt_z;

//     Eigen::Vector3d l = goal - pf;
//     Eigen::Vector3d vf = (v_max / l.norm()) * l;
//     vf(2) = 0; // % Note: 0 maybe better, for the p curve wont go down to meet the vf

//     Eigen::Vector3d af = Eigen::Vector3d::Zero();

//     // % Choose the time as running in average velocity
//     // double decay_parameter = 0.5;
//     // double T = 0.2;
    
//     double j_limit = 4;
//     double a_limit = 4;
//     double v_limit = 2;

//     double T1 = fabs(af(0)-a0(0))/j_limit > fabs(af(1)-a0(1))/j_limit ? fabs(af(0)-a0(0))/j_limit : fabs(af(1)-a0(1))/j_limit;
//     T1 = T1 > fabs(af(2)-a0(2))/j_limit ? T1 : fabs(af(2)-a0(2))/j_limit;
//     double T2 = fabs(vf(0)-v0(0))/a_limit > fabs(vf(1)-v0(1))/a_limit ? fabs(vf(0)-v0(0))/a_limit : fabs(vf(1)-v0(1))/a_limit;
//     T2 = T2 > fabs(vf(2)-v0(2))/a_limit ? T2 : fabs(vf(2)-v0(2))/a_limit;
//     double T3 = fabs(pf(0)-p0(0))/v_limit > fabs(pf(1)-p0(1))/v_limit ? fabs(pf(0)-p0(0))/v_limit : fabs(pf(1)-p0(1))/v_limit;
//     T3 = T3 > fabs(pf(2)-p0(2))/v_limit ? T3 : fabs(pf(2)-p0(2))/v_limit;

//     double T = T1 > T2 ? T1 : T2;
//     T = T > T3 ? T : T3;
//     T = T < 0.5 ? 0.5 : T;

//     int times = T / delt_t;

//     // Show
//     // for(int i=0; i<3; i++)
//     // {
//     //     ROS_INFO("P0, %lf, %lf, %lf", p0(0), p0(1), p0(2));
//     //     ROS_INFO("V0, %lf, %lf, %lf", v0(0), v0(1), v0(2));
//     //     ROS_INFO("A0, %lf, %lf, %lf", a0(0), a0(1), a0(2));

//     //     ROS_INFO("Pf, %lf, %lf, %lf", pf(0), pf(1), pf(2));
//     //     ROS_INFO("Vf, %lf, %lf, %lf", vf(0), vf(1), vf(2));
//     //     ROS_INFO("Af, %lf, %lf, %lf", af(0), af(1), af(2));

//     //     ROS_INFO("T, %lf, yaw0, %lf, v_max, %lf, d, %lf", T, yaw0, v_max, d);
//     // }

//     p = Eigen::MatrixXd::Zero(times, 3);
//     v = Eigen::MatrixXd::Zero(times, 3);
//     a = Eigen::MatrixXd::Zero(times, 3);
//     t = Eigen::VectorXd::Zero(times);

//     // // % calculate optimal jerk controls by Mark W. Miller
//     // for(int ii=0; ii<3; ii++)
//     // {
//     //     double delt_a = af(ii) - a0(ii);
//     //     double delt_v = vf(ii) - v0(ii) - a0(ii)*T;
//     //     double delt_p = pf(ii) - p0(ii) - v0(ii)*T - 0.5*a0(ii)*T*T;

//     //     //%  if vf is not free
//     //     double alpha = delt_a*60/pow(T,3) - delt_v*360/pow(T,4) + delt_p*720/pow(T,5);
//     //     double beta = -delt_a*24/pow(T,2) + delt_v*168/pow(T,3) - delt_p*360/pow(T,4);
//     //     double gamma = delt_a*3/T - delt_v*24/pow(T,2) + delt_p*60/pow(T,3);

//     //     for(int jj=0; jj<times; jj++)
//     //     {
//     //         double tt = (jj + 1)*delt_t;
//     //         t(jj) = tt;
//     //         p(jj,ii) = alpha/120*pow(tt,5) + beta/24*pow(tt,4) + gamma/6*pow(tt,3) + a0(ii)/2*pow(tt,2) + v0(ii)*tt + p0(ii);
//     //         v(jj,ii) = alpha/24*pow(tt,4) + beta/6*pow(tt,3) + gamma/2*pow(tt,2) + a0(ii)*tt + v0(ii);
//     //         a(jj,ii) = alpha/6*pow(tt,3) + beta/2*pow(tt,2) + gamma*tt + a0(ii);

//     //         // Only for checking acceleration
//     //         if (a(jj, ii) > planning_max_accel(ii)) planning_max_accel(ii) = a(jj, ii);
//     //     }
//     // }

//     bool limit_check = false;
//     Eigen::Vector3d planning_max_accel = Eigen::Vector3d::Zero();
//     Eigen::Vector3d planning_max_vel = Eigen::Vector3d::Zero();
//     int max_ite_num = 8;
//     int ite_num = 0;
//     auto start = std::chrono::steady_clock::now();

//     while (!limit_check && ite_num++ < max_ite_num) {
//         // Only for checking acceleration
//         planning_max_accel = Eigen::Vector3d::Zero();
//         planning_max_vel = Eigen::Vector3d::Zero();
//         limit_check = true;

//         // % calculate optimal jerk controls by Mark W. Miller
//         for(int ii=0; ii<3; ii++)
//         {
//             double delt_a = af(ii) - a0(ii);
//             double delt_v = vf(ii) - v0(ii) - a0(ii)*T;
//             double delt_p = pf(ii) - p0(ii) - v0(ii)*T - 0.5*a0(ii)*T*T;

//             // % if vf is not free
//             double alpha = delt_a*60/pow(T,3) - delt_v*360/pow(T,4) + delt_p*720/pow(T,5);
//             double beta = -delt_a*24/pow(T,2) + delt_v*168/pow(T,3) - delt_p*360/pow(T,4);
//             double gamma = delt_a*3/T - delt_v*24/pow(T,2) + delt_p*60/pow(T,3);

//             for(int jj=0; jj<times; jj++)
//             {
//                 double tt = (jj + 1)*delt_t;
//                 t(jj) = tt;
//                 p(jj,ii) = alpha/120*pow(tt,5) + beta/24*pow(tt,4) + gamma/6*pow(tt,3) + a0(ii)/2*pow(tt,2) + v0(ii)*tt + p0(ii);
//                 v(jj,ii) = alpha/24*pow(tt,4) + beta/6*pow(tt,3) + gamma/2*pow(tt,2) + a0(ii)*tt + v0(ii);
//                 a(jj,ii) = alpha/6*pow(tt,3) + beta/2*pow(tt,2) + gamma*tt + a0(ii);

//                 // Only for checking acceleration
//                 if (fabs(a(jj, ii)) > planning_max_accel(ii)) planning_max_accel(ii) = fabs(a(jj, ii));
//                 if (fabs(v(jj, ii)) > planning_max_vel(ii)) planning_max_vel(ii) = fabs(v(jj, ii));
//             }
//         }

//         ROS_WARN("Max acceleration: %lf, %lf, %lf", planning_max_accel(0), planning_max_accel(1), planning_max_accel(2));
//         ROS_WARN("Theta V is: %lf", theta_v);
//         ROS_WARN("az is : %lf", a0(2));
        
//         for (int i = 0; i < 3; ++i) {
//             if (planning_max_accel(i) > a_limit/* || planning_max_vel(i) > v_limit*/) {
//                 limit_check = false;

//                 // Increase total time by 10% to decrease max acceleration
//                 T *= 1.1;
//                 times = T / delt_t;
//                 p = Eigen::MatrixXd::Zero(times, 3);
//                 v = Eigen::MatrixXd::Zero(times, 3);
//                 a = Eigen::MatrixXd::Zero(times, 3);
//                 t = Eigen::VectorXd::Zero(times);

//                 break;
//             }
//         }
//     }

//     // Check calculation time
//     auto end = std::chrono::steady_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     ROS_WARN("primitives time: %lf ms", duration.count()/(double)1000.0);
//     ROS_WARN("Iteration times: %d", ite_num);

//     // Only for checking acceleration
//     geometry_msgs::Vector3 max_accel;
//     max_accel.x = planning_max_accel(0);
//     max_accel.y = planning_max_accel(1);
//     max_accel.z = planning_max_accel(2);

//     max_accel_pub.publish(max_accel);
// }

bool motion_primitives_with_table(Eigen::Vector3d p0, Eigen::Vector3d v0, Eigen::Vector3d a0, double yaw0, double theta_h,
                                  double theta_v, Eigen::Vector3d goal, double d, double v_max, double delt_t,
                                  Eigen::MatrixXd &p, Eigen::MatrixXd &v, Eigen::MatrixXd &a, Eigen::VectorXd &t)
{
    double delt_x = d*cos(theta_v)*cos(theta_h+yaw0);
    double delt_y = d*cos(theta_v)*sin(theta_h+yaw0);
    double delt_z = d*sin(theta_v);

    Eigen::Vector3d pf;
    pf << p0(0)+delt_x, p0(1)+delt_y, p0(2)+delt_z;

    Eigen::Vector3d l = goal - pf;
    Eigen::Vector3d vf = (v_max / l.norm()) * l;
    vf(2) = 0; // % Note: 0 maybe better, for the p curve wont go down to meet the vf

    Eigen::Vector3d af = Eigen::Vector3d::Zero();
    
    double j_limit = 4;
    double a_limit = 3;
    double v_limit = 2;

    double T1, T2, T3, T;

    if (v0.cwiseAbs().maxCoeff() > v_limit || a0.cwiseAbs().maxCoeff() > a_limit || !table_initialized) {
        T1 = fabs(af(0)-a0(0))/j_limit > fabs(af(1)-a0(1))/j_limit ? fabs(af(0)-a0(0))/j_limit : fabs(af(1)-a0(1))/j_limit;
        T1 = T1 > fabs(af(2)-a0(2))/j_limit ? T1 : fabs(af(2)-a0(2))/j_limit;

        T2 = fabs(vf(0)-v0(0))/a_limit > fabs(vf(1)-v0(1))/a_limit ? fabs(vf(0)-v0(0))/a_limit : fabs(vf(1)-v0(1))/a_limit;
        T2 = T2 > fabs(vf(2)-v0(2))/a_limit ? T2 : fabs(vf(2)-v0(2))/a_limit;

        T3 = fabs(pf(0)-p0(0))/v_limit > fabs(pf(1)-p0(1))/v_limit ? fabs(pf(0)-p0(0))/v_limit : fabs(pf(1)-p0(1))/v_limit;
        T3 = T3 > fabs(pf(2)-p0(2))/v_limit ? T3 : fabs(pf(2)-p0(2))/v_limit;
    } else {
        T1 = table->query_pva_table(delt_x, v0(0), vf(0), a0(0));
        T2 = table->query_pva_table(delt_y, v0(0), vf(0), a0(0));
        T3 = table->query_pva_table(delt_z, v0(0), vf(0), a0(0));

        ROS_ERROR("Table used");

        if (T1 == -1 || T2 == -1 || T3 == -1) return false;
    }

    T = T1 > T2 ? T1 : T2;
    T = T > T3 ? T : T3;
    T = T < 0.5 ? 0.5 : T;
    
    int times = T / delt_t;

    // Show
    // for(int i=0; i<3; i++)
    // {
    //     ROS_INFO("P0, %lf, %lf, %lf", p0(0), p0(1), p0(2));
    //     ROS_INFO("V0, %lf, %lf, %lf", v0(0), v0(1), v0(2));
    //     ROS_INFO("A0, %lf, %lf, %lf", a0(0), a0(1), a0(2));

    //     ROS_INFO("Pf, %lf, %lf, %lf", pf(0), pf(1), pf(2));
    //     ROS_INFO("Vf, %lf, %lf, %lf", vf(0), vf(1), vf(2));
    //     ROS_INFO("Af, %lf, %lf, %lf", af(0), af(1), af(2));

    //     ROS_INFO("T, %lf, yaw0, %lf, v_max, %lf, d, %lf", T, yaw0, v_max, d);
    // }

    p = Eigen::MatrixXd::Zero(times, 3);
    v = Eigen::MatrixXd::Zero(times, 3);
    a = Eigen::MatrixXd::Zero(times, 3);
    t = Eigen::VectorXd::Zero(times);

    Eigen::Vector3d planning_max_accel = Eigen::Vector3d::Zero();
    Eigen::Vector3d planning_max_vel = Eigen::Vector3d::Zero();
    auto start = std::chrono::steady_clock::now();

    // Only for checking acceleration
    planning_max_accel = Eigen::Vector3d::Zero();
    planning_max_vel = Eigen::Vector3d::Zero();

    // % calculate optimal jerk controls by Mark W. Miller
    for(int ii=0; ii<3; ii++)
    {
        double delt_a = af(ii) - a0(ii);
        double delt_v = vf(ii) - v0(ii) - a0(ii)*T;
        double delt_p = pf(ii) - p0(ii) - v0(ii)*T - 0.5*a0(ii)*T*T;

        // % if vf is not free
        double alpha = delt_a*60/pow(T,3) - delt_v*360/pow(T,4) + delt_p*720/pow(T,5);
        double beta = -delt_a*24/pow(T,2) + delt_v*168/pow(T,3) - delt_p*360/pow(T,4);
        double gamma = delt_a*3/T - delt_v*24/pow(T,2) + delt_p*60/pow(T,3);

        for(int jj=0; jj<times; jj++)
        {
            double tt = (jj + 1)*delt_t;
            t(jj) = tt;
            p(jj,ii) = alpha/120*pow(tt,5) + beta/24*pow(tt,4) + gamma/6*pow(tt,3) + a0(ii)/2*pow(tt,2) + v0(ii)*tt + p0(ii);
            v(jj,ii) = alpha/24*pow(tt,4) + beta/6*pow(tt,3) + gamma/2*pow(tt,2) + a0(ii)*tt + v0(ii);
            a(jj,ii) = alpha/6*pow(tt,3) + beta/2*pow(tt,2) + gamma*tt + a0(ii);

            // Only for checking acceleration
            if (fabs(a(jj, ii)) > planning_max_accel(ii)) planning_max_accel(ii) = fabs(a(jj, ii));
            if (fabs(v(jj, ii)) > planning_max_vel(ii)) planning_max_vel(ii) = fabs(v(jj, ii));
        }
    }

    ROS_WARN("Max acceleration: %lf, %lf, %lf", planning_max_accel(0), planning_max_accel(1), planning_max_accel(2));
    ROS_WARN("Theta V is: %lf", theta_v);
    ROS_WARN("az is : %lf", a0(2));

    // Check calculation time
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    ROS_WARN("primitives time: %lf ms", duration.count()/(double)1000.0);

    // Only for checking acceleration
    geometry_msgs::Vector3 max_accel;
    max_accel.x = planning_max_accel(0);
    max_accel.y = planning_max_accel(1);
    max_accel.z = planning_max_accel(2);

    max_accel_pub.publish(max_accel);

    return true;
}

/* Publish markers to show the path in rviz */
void marker_publish(Eigen::MatrixXd &Points) 
{
    visualization_msgs::Marker points, line_strip;
    points.header.frame_id = line_strip.header.frame_id = "world";
    points.header.stamp = line_strip.header.stamp = ros::Time::now();
    points.action = line_strip.action = visualization_msgs::Marker::ADD;
    points.ns = line_strip.ns = "points_and_lines";

    points.id = 0;
    line_strip.id = 1;

    points.type = visualization_msgs::Marker::POINTS;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;

    points.scale.x = 0.2;
    points.scale.y = 0.2;

    // Line width
    line_strip.scale.x = 0.1; 

    // Points are green
    points.color.g = 1.0;
    points.color.a = 1.0;

    // Line strip is blue
    line_strip.color.b = 1.0;
    line_strip.color.a = 1.0;

    line_strip.lifetime = ros::Duration(0);

    int point_num = Points.rows();

    for(int i=0; i<point_num; i++)
    {
        geometry_msgs::Point p;
        p.x = Points(i, 0);
        p.y = Points(i, 1);
        p.z = Points(i, 2);

        // ROS_INFO("p.x %lf", p.x);

        points.points.push_back(p);
        line_strip.points.push_back(p);
    }

    current_marker_pub.publish(points);
    current_marker_pub.publish(line_strip);
}

// void trajectoryCallback(Eigen::Vector3d &p_goal, Eigen::Vector3d &p0, Eigen::Vector3d &v0, Eigen::Vector3d &a0, double &yaw0) 
void trajectoryCallback(const ros::TimerEvent& e) {
    if(!state_updating)
    {
        // To do: Update p_goal in accordance with keyboard/joy command

        state_locked = true;

        // TODO: Safety strategy for emergency stop should be added here

        // geometry_msgs::Pose traj_pt;
        trajectory_msgs::JointTrajectoryPoint pva_setpoint;
        pva_setpoint.positions.push_back(p_store(0));
        pva_setpoint.positions.push_back(p_store(1));
        pva_setpoint.positions.push_back(flight_altitude);
        pva_setpoint.positions.push_back(yaw_store);

        pva_setpoint.velocities.push_back(0.0);
        pva_setpoint.velocities.push_back(0.0);
        pva_setpoint.velocities.push_back(0.0);

        pva_setpoint.accelerations.push_back(0.0);
        pva_setpoint.accelerations.push_back(0.0);
        pva_setpoint.accelerations.push_back(0.0);

        if (uav_pause) {
            pva_setpoint.positions[0] = p_store(0);
            pva_setpoint.positions[1] = p_store(1);
            pva_setpoint.positions[2] = flight_altitude;
            pva_setpoint.positions[3] = yaw_store;

            pva_setpoint.velocities[0] = 0.0;
            pva_setpoint.velocities[1] = 0.0;
            pva_setpoint.velocities[2] = 0.0;

            pva_setpoint.accelerations[0] = 0.0;
            pva_setpoint.accelerations[1] = 0.0;
            pva_setpoint.accelerations[2] = 0.0;

            traj_point_pub.publish(pva_setpoint);

            // traj_pt.position.x = p_store(0);
            // traj_pt.position.y = p_store(1);
            // // traj_pt.position.z = p_store(2);  // Added on July 27
            // traj_pt.position.z = 1.5;
            // traj_pt.orientation.x = yaw_store;

            // for (int i = 0; i < 3; ++i) {
            //     traj_point_pub.publish(traj_pt);
            // }

            ROS_INFO("Pause mode");
            ROS_INFO("Stay still");

            state_locked = false;

            return;
        }

        if (in_safety_mode && !safety_mode_recover) {
            pva_setpoint.positions[0] = p_store(0);
            pva_setpoint.positions[1] = p_store(1);
            pva_setpoint.positions[2] = p_store(2);
            pva_setpoint.positions[3] = yaw_store;

            pva_setpoint.velocities[0] = 0.0;
            pva_setpoint.velocities[1] = 0.0;
            pva_setpoint.velocities[2] = 0.0;

            pva_setpoint.accelerations[0] = 0.0;
            pva_setpoint.accelerations[1] = 0.0;
            pva_setpoint.accelerations[2] = 0.0;

            traj_point_pub.publish(pva_setpoint);


            // traj_pt.position.x = p_store(0);
            // traj_pt.position.y = p_store(1);
            // traj_pt.position.z = p_store(2);
            // traj_pt.orientation.x = yaw_store;

            // for (int i = 0; i < 3; ++i) {
            //     traj_point_pub.publish(traj_pt);
            // }

            ROS_INFO("Safety mode");
            ROS_INFO("Please trigger recovering process");

            // if (test_angle) {
            //     ROS_INFO("Angle is too large");
            //     ROS_INFO("yaw %lf", yaw0 / M_PI * 180.0);
            //     ROS_INFO("theta %lf", th_h / M_PI * 180.0);
            //     ROS_INFO("phi %lf", ph_h / M_PI * 180.0);            
            // }

            state_locked = false;

            return;
        }

        // Reset the flags
        in_safety_mode = false;
        safety_mode_recover = false;

        ROS_INFO("rc_theta: %lf", rc_theta / M_PI * 180.0);
        ROS_INFO("p_goal: %lf, %lf, %lf", p_goal(0), p_goal(1), p_goal(2));

        /** Moition primitives **/
        Eigen::Vector3d delt_p = p_goal - p0;
        double phi_h = atan2(delt_p(1), delt_p(0)); //% horizental offset angle
        double phi_v = atan2(delt_p(2), sqrt(delt_p(0) * delt_p(0) + delt_p(1) * delt_p(1))); //% vertical offset angle

        // %calculate cost for sampled points
        Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(ANGLE_H_NUM * ANGLE_V_NUM, 4);
        double theta_h = 0;
        double theta_v = 0;

        for(int i=0; i<ANGLE_H_NUM; i++)
        {
            for(int j=0; j<ANGLE_V_NUM; j++)
            {
                theta_h = Angle_h(i);
                theta_v = Angle_v(j);
                int m = i*ANGLE_V_NUM + j; //sequence number
                // Vectorial angle can never be larger than PI
                double goal_diff = fabs(yaw0+theta_h-phi_h) > M_PI ? 2*M_PI - fabs(yaw0+theta_h-phi_h) : yaw0+theta_h-phi_h;

                cost(m, 0) = pp.k1_xy*goal_diff*goal_diff + pp.k1_z*(theta_v-phi_v)*(theta_v-phi_v) +
                        pp.k2_xy*(theta_h-theta_h_last)*(theta_h-theta_h_last) + pp.k2_z*(theta_v-theta_v_last)*(theta_v-theta_v_last) +
                        pp.k3*F_cost(i,j);
                cost(m, 1) = theta_h;
                cost(m, 2) = theta_v;
                cost(m, 3) = (1-F_cost(i,j)) * pp.d_ref;
            }
        }

        //% Rank by cost, small to large
        for(int m=0; m<ANGLE_H_NUM*ANGLE_V_NUM-1; m++)
        {
            for(int n=0; n<ANGLE_H_NUM*ANGLE_V_NUM-m-1; n++)
            {
                if(cost(n,0) > cost(n+1,0))
                {
                    Eigen::Vector4d temp = cost.row(n+1);
                    cost.row(n+1) = cost.row(n);
                    cost.row(n) = temp;
                }
            }
        }

        //% max velocity is decreased concerning current velocity direction and goal
        //% direction
        double v_scale = std::max(delt_p.dot(v0)/v0.norm()/delt_p.norm(), pp.v_scale_min);
        // If v0.norm() = 0, v_scale = nan
        v_scale = (v0.norm() == 0) ? pp.v_scale_min : v_scale;
        v_scale = v_scale > 1 ? 1 : v_scale;
        double v_max = pp.v_max_ori * v_scale;

        bool flag = false;

        for(int seq=0; seq<pp.max_plan_num; seq++)
        {
            Eigen::MatrixXd p;
            Eigen::MatrixXd v;
            Eigen::MatrixXd a;
            Eigen::VectorXd t;
            
            // motion_primitives(p0, v0, a0, yaw0, cost(seq,1), cost(seq,2), p_goal, cost(seq,3), v_max, pp.delt_t, p, v, a, t);
            bool is_feasible = motion_primitives_with_table(p0, v0, a0, yaw0, cost(seq,1), cost(seq,2), p_goal, cost(seq,3), v_max, pp.delt_t, p, v, a, t);
            if (!is_feasible) {
                ROS_WARN("Trajectory does not have any feasible T!");
                continue;
            }

            p.col(2) = Eigen::VectorXd::Ones(p.rows())*flight_altitude;
            v.col(2) = Eigen::VectorXd::Zero(p.rows());
            a.col(2) = Eigen::VectorXd::Zero(p.rows());

            const int Num = p.rows(); // get points number on the path
            Eigen::Vector3f *sim_traj = new Eigen::Vector3f[Num];

            for (int i = 0; i < Num; ++i) {
                sim_traj[i](0) = (float)p.row(i)(0);
                sim_traj[i](1) = (float)p.row(i)(1);
                sim_traj[i](2) = (float)p.row(i)(2);
            }

            flag = rrb.collision_checking(sim_traj, Num, 0.6); // collision_checking

            if(flag) 
            {
                ROS_INFO("traj_safe");
                theta_h_last = cost(seq,1); // Update last theta
                theta_v_last = cost(seq,2);
                ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);
                // ROS_WARN("angle: %lf", cost(seq,1) / M_PI * 180.0);

                // Publish down sampled path points
                const int point_num_pub = 5;
                int interval_num = Num / point_num_pub;
                if (interval_num > 0)
                {
                    Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(point_num_pub+1, 3);
                    for(int pubi = 0; pubi < point_num_pub; pubi++)
                    {
                        show_points.row(pubi) = p.row(pubi*interval_num);
                    }
                    show_points.row(point_num_pub) = p.row(Num-1);
                    marker_publish(show_points);
                }
                
                // Aggressive/Conservative strategy according to entropy
                float entropy2d = rrb.compute_entropy2d(x_centre, y_centre, z_centre);
                float entropy1d = rrb.compute_entropy1d(x_centre, y_centre, z_centre);
                ROS_INFO("entropy2d %f", entropy2d);
                ROS_INFO("entropy1d %f", entropy1d);

                int control_point = ceil(Num / (entropy2d + 2.0));
                control_point = control_point > Num-1 ? Num-1 : control_point;
                // ROS_INFO("control point %d", control_point);

                control_point = ceil(Num / 10.0);
                control_point = control_point < 5 ? 5 : control_point;
                // control_point = 10;
                control_point = control_point > Num-1 ? Num-1 : control_point;

                std_msgs::Int16 control_p_idx;
                control_p_idx.data = control_point;
                control_p_idx_pub.publish(control_p_idx);

                pva_setpoint.positions[0] = p.row(control_point)(0);
                pva_setpoint.positions[1] = p.row(control_point)(1);
                pva_setpoint.positions[2] = p.row(control_point)(2);
                double desired_yaw = atan2(v.row(control_point)(1) - p_store(1), v.row(control_point)(0) - p_store(0));
                double yaw_diff = desired_yaw - yaw_store;

                double yaw_control_coef = 0.05;
                while (yaw_diff > M_PI) yaw_diff -= 2*M_PI;
                while (yaw_diff < -M_PI) yaw_diff += 2*M_PI;
                // Normalized to (-1, 1)
                yaw_diff /= M_PI;
                yaw_diff *= yaw_control_coef;

                pva_setpoint.positions[3] = yaw_store + yaw_diff;
                // pva_setpoint.positions[3] = yaw_store;

                pva_setpoint.velocities[0] = v.row(control_point)(0);
                pva_setpoint.velocities[1] = v.row(control_point)(1);
                pva_setpoint.velocities[2] = v.row(control_point)(2);

                pva_setpoint.accelerations[0] = a.row(control_point)(0);
                pva_setpoint.accelerations[1] = a.row(control_point)(1);
                pva_setpoint.accelerations[2] = a.row(control_point)(2);

                // traj_pt.position.x = p.row(control_point)(0);
                // traj_pt.position.y = p.row(control_point)(1);
                // traj_pt.position.z = p.row(control_point)(2);
                // traj_pt.orientation.x = atan2(v.row(control_point)(1), v.row(control_point)(0));

                bool dense_env_flag = entropy2d > 0.5;

                // Check whether the orientation of traj is in FOV
                if (fabs(cost(seq,1)) >= M_PI/3 && dense_env_flag) {
                    // Stop

                    p_store = p0;
                    yaw_store = yaw0 + cost(seq,1);

                    pva_setpoint.positions[0] = p_store(0);
                    pva_setpoint.positions[1] = p_store(1);
                    pva_setpoint.positions[2] = p_store(2);
                    pva_setpoint.positions[3] = yaw_store;

                    pva_setpoint.velocities[0] = 0.0;
                    pva_setpoint.velocities[1] = 0.0;
                    pva_setpoint.velocities[2] = 0.0;

                    pva_setpoint.accelerations[0] = 0.0;
                    pva_setpoint.accelerations[1] = 0.0;
                    pva_setpoint.accelerations[2] = 0.0;

                    traj_point_pub.publish(pva_setpoint);

                    // traj_pt.position.x = p_store(0);
                    // traj_pt.position.y = p_store(1);
                    // traj_pt.position.z = p_store(2);
                    // traj_pt.orientation.x = yaw_store;

                    // for (int i = 0; i < 3; ++i) {
                    //     traj_point_pub.publish(traj_pt);
                    // }
                    
                    // coeff = cost(seq,1) >= M_PI/2 ? 1 : -1;
                    // twist_angular = coeff * certain_rate;

                    ROS_INFO("Stay still and rotate");
                    ROS_INFO("Vertical %lf", cost(seq, 1)*180/M_PI);

                    Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(6, 3);
                    for(int pubi = 0; pubi < 6; pubi++)
                    {
                        show_points(pubi, 0) = p_store(0);
                        show_points(pubi, 1) = p_store(1);
                        show_points(pubi, 2) = p_store(2);
                    }
                    marker_publish(show_points);
                } else {
                    if (fabs(cost(seq,1)) >= M_PI/2) {
                        // Stop
                        in_safety_mode = true;

                        ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");
                        // ROS_WARN("Too large!");

                        p_store = p0;
                        yaw_store = yaw0 + cost(seq,1);

                        pva_setpoint.positions[0] = p_store(0);
                        pva_setpoint.positions[1] = p_store(1);
                        pva_setpoint.positions[2] = p_store(2);
                        pva_setpoint.positions[3] = yaw_store;

                        pva_setpoint.velocities[0] = 0.0;
                        pva_setpoint.velocities[1] = 0.0;
                        pva_setpoint.velocities[2] = 0.0;

                        pva_setpoint.accelerations[0] = 0.0;
                        pva_setpoint.accelerations[1] = 0.0;
                        pva_setpoint.accelerations[2] = 0.0;

                        traj_point_pub.publish(pva_setpoint);

                        // traj_pt.position.x = p_store(0);
                        // traj_pt.position.y = p_store(1);
                        // traj_pt.position.z = p_store(2);
                        // traj_pt.orientation.x = yaw_store;

                        // // test_angle = true;
                        // // th_h = cost(seq, 1);
                        // // ph_h = phi_h;

                        // for (int i = 0; i < 3; ++i) {
                        //     traj_point_pub.publish(traj_pt);
                        // }
                    
                        // coeff = cost(seq,1) >= M_PI/2 ? 1 : -1;
                        // twist_angular = coeff * certain_rate;

                        ROS_INFO("Stay still and rotate");
                        ROS_INFO("Vertical %lf", cost(seq, 1)*180/M_PI);

                        ROS_WARN("Trapped in safety mode!");
                        
                        Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(6, 3);
                        for(int pubi = 0; pubi < 6; pubi++)
                        {
                            show_points(pubi, 0) = p_store(0);
                            show_points(pubi, 1) = p_store(1);
                            show_points(pubi, 2) = p_store(2);
                        }
                        marker_publish(show_points);
                    } else {
                        traj_point_pub.publish(pva_setpoint);
                        // traj_point_pub.publish(traj_pt);
                        // // twist_angular = 0.0;
                    }
                }
                break;
            } 
            else 
            {
                ROS_INFO("traj_unsafe");
            }
        }

        if(!flag){
            ROS_WARN("No valid trajectory found!");

            ROS_WARN("No traj!");
            // ROS_WARN("No traj!");
            // ROS_WARN("No traj!");
            // ROS_WARN("No traj!");
            // ROS_WARN("No traj!");
            // ROS_WARN("No traj!");
            // ROS_WARN("No traj!");
            // ROS_WARN("No traj!");

            ROS_WARN("Trapped in safety mode!");
            in_safety_mode = true;

            //emergency stop?      4th July
            p_store = p0;
            yaw_store = yaw0;

            pva_setpoint.positions[0] = p_store(0);
            pva_setpoint.positions[1] = p_store(1);
            pva_setpoint.positions[2] = p_store(2);
            pva_setpoint.positions[3] = yaw_store;

            pva_setpoint.velocities[0] = 0.0;
            pva_setpoint.velocities[1] = 0.0;
            pva_setpoint.velocities[2] = 0.0;

            pva_setpoint.accelerations[0] = 0.0;
            pva_setpoint.accelerations[1] = 0.0;
            pva_setpoint.accelerations[2] = 0.0;

            traj_point_pub.publish(pva_setpoint);
        
            // traj_pt.position.x = p_store(0);
            // traj_pt.position.y = p_store(1);
            // traj_pt.position.z = p_store(2);
            // traj_pt.orientation.x = yaw_store;

            // for (int i = 0; i < 3; ++i) {
            //     traj_point_pub.publish(traj_pt);
            // }

            Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(6, 3);
            for(int pubi = 0; pubi < 6; pubi++)
                {
                    show_points(pubi, 0) = p_store(0);
                    show_points(pubi, 1) = p_store(1);
                    show_points(pubi, 2) = p_store(2);
                }
            marker_publish(show_points);
        }

        double algo_time = ros::Time::now().toSec() - _algorithm_time.toSec();
        ROS_INFO("algorithm time is: %lf", algo_time);
        _algorithm_time = ros::Time::now();

        // TODO: Publish the control values

        state_locked = false;
    }

}

/*void stateCallBack(const nav_msgs::OdometryConstPtr& odom, const sensor_msgs::ImuConstPtr& imu)
{
    ROS_INFO("State data!");
    Eigen::Vector3d p_goal;
    p_goal << 10, 0, 2;
    Eigen::Vector3d p0;
    p0 << odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z; 
    Eigen::Vector3d v0;
    v0 << odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z; 
    Eigen::Vector3d a0;
    a0 << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;

    double x = imu->orientation.x; 
    double y = imu->orientation.y; 
    double z = imu->orientation.z; 
    double w = imu->orientation.w; 
    double yaw0 = atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z);
    // ROS_WARN("yaw0 is: %lf", yaw0);

    trajectoryCallback(p_goal, p0, v0, a0, yaw0);
}*/

void odomCallback(const nav_msgs::OdometryConstPtr& odom)
{
    if(!state_locked)
    {
        state_updating = true;

        p0(0) = odom->pose.pose.position.x;
        p0(1) = odom->pose.pose.position.y;
        p0(2) = odom->pose.pose.position.z;

        v0(0) = odom->twist.twist.linear.x;
        v0(1) = odom->twist.twist.linear.y;
        v0(2) = odom->twist.twist.linear.z;

        if(!in_safety_mode) {
            // printf("p0: %f %f %f\n", p0(0), p0(1), p0(2));
            p_store = p0;
            yaw_store = yaw0;
        }

        state_updating = false;
    }
    odom_initilized = true;
}

void imuCallback(const sensor_msgs::ImuConstPtr& imu)
{
    if(!state_locked)
    {
        state_updating = true;

        a0(0) = imu->linear_acceleration.x;
        a0(1) = imu->linear_acceleration.y;
        a0(2) = imu->linear_acceleration.z - GRAVATY;
        // a0(2) = imu->linear_acceleration.z;

        double x = imu->orientation.x; 
        double y = imu->orientation.y; 
        double z = imu->orientation.z; 
        double w = imu->orientation.w; 
        yaw0 = atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z);

        // ROS_WARN("IMU: %lf", imu->linear_acceleration.z);
        // ROS_WARN("Acceleration Z: %lf", a0(2));

        state_updating = false;
    }
    imu_initilized = true;
}

void pGoalCallback(const geometry_msgs::Twist& msg)
{
    rc_theta += msg.angular.z * M_PI / 8.0;
    ROS_INFO("rc_theta: %lf", rc_theta / M_PI * 180.0);

    geometry_msgs::Quaternion p_goal_quat = tf::createQuaternionMsgFromYaw(rc_theta);
    
    geometry_msgs::PoseStamped p_goal_pose;
    p_goal_pose.header.stamp = ros::Time::now();
    p_goal_pose.header.frame_id = "world";
    p_goal_pose.pose.position.x = p0(0);
    p_goal_pose.pose.position.y = p0(1);
    p_goal_pose.pose.position.z = p0(2);
    p_goal_pose.pose.orientation = p_goal_quat;

    p_goal_pose_pub.publish(p_goal_pose);

    if (!state_locked && fabs(msg.angular.z) > 0.5) {
        p_goal(0) = x_centre + p_goal_radius * cos(rc_theta);
        p_goal(1) = y_centre + p_goal_radius * sin(rc_theta);
        p_goal(2) = flight_altitude;
    }

    // Press 'p' to pause/unpause
    if (msg.angular.x > 0.5) {
        uav_pause = false;
    } else if (fabs(msg.angular.x) < 0.5) {
        uav_pause = true;
    } else {
        if (!in_safety_mode) {
            ROS_WARN("No need to recover!");
            return;
        }

        // Resetting flight altitude
        // geometry_msgs::Pose traj_pt;
        trajectory_msgs::JointTrajectoryPoint pva_setpoint;
        p_store(0) = p0(0);
        p_store(1) = p0(1);
        p_store(2) = flight_altitude;
        yaw_store = yaw0;

        pva_setpoint.positions.push_back(p_store(0));
        pva_setpoint.positions.push_back(p_store(1));
        pva_setpoint.positions.push_back(p_store(2));
        pva_setpoint.positions.push_back(yaw_store);

        pva_setpoint.velocities.push_back(0.0);
        pva_setpoint.velocities.push_back(0.0);
        pva_setpoint.velocities.push_back(0.0);

        pva_setpoint.accelerations.push_back(0.0);
        pva_setpoint.accelerations.push_back(0.0);
        pva_setpoint.accelerations.push_back(0.0);

        traj_point_pub.publish(pva_setpoint);

        // traj_pt.position.x = p_store(0);
        // traj_pt.position.y = p_store(1);
        // traj_pt.position.z = p_store(2);
        // traj_pt.orientation.x = yaw_store;

        // for (int i = 0; i < 3; ++i) {
        //     traj_point_pub.publish(traj_pt);
        // }

        rc_theta = yaw0; // When on px4, rc_theta should be reset to 0.0?
        ROS_INFO("rc_theta: %lf", rc_theta / M_PI * 180.0);

        // Resetting p_goal
        p_goal(0) = x_centre + p_goal_radius * cos(rc_theta);
        p_goal(1) = y_centre + p_goal_radius * sin(rc_theta);
        p_goal(2) = flight_altitude;

        p_goal_quat = tf::createQuaternionMsgFromYaw(rc_theta);
    
        p_goal_pose.header.stamp = ros::Time::now();
        p_goal_pose.header.frame_id = "world";
        p_goal_pose.pose.position.x = p0(0);
        p_goal_pose.pose.position.y = p0(1);
        p_goal_pose.pose.position.z = p0(2);
        p_goal_pose.pose.orientation = p_goal_quat;

        p_goal_pose_pub.publish(p_goal_pose);

        safety_mode_recover = true;
        uav_pause = true;
    }
}

void visCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_env(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dis_field(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*cloud, *cloud_env);

    for (int i = 0; i < cloud_env->points.size(); ++i) {
        if (cloud_env->points[i].z < -0.22) {
            cloud_env->points[i].r = 0;
            cloud_env->points[i].g = 255;
            cloud_env->points[i].b = 0;
        } else {
            cloud_env->points[i].r = 255;
            cloud_env->points[i].g = 0;
            cloud_env->points[i].b = 0;
        }
    }

    double trunc_distance = 1.0; // Truncation distance is 1.0
    double boundary = pow(2.0, POW) * resolution / 2; // Here the boundary is 6.4m

    for (double vis_x = resolution/2; vis_x < boundary; vis_x += resolution) {
        for (double vis_y = resolution/2; vis_y < boundary; vis_y += resolution) {
            // Skip the truncation area
            if (vis_x <= trunc_distance && vis_y <= trunc_distance) {
                continue;
            }

            pcl::PointXYZRGB vis_point;
            int dir_x[4] = {1, 1, -1, -1};
            int dir_y[4] = {1, -1, 1, -1};

            for (int i = 0; i < 4; ++i) {
                vis_point.x = x_centre + vis_x * dir_x[i];
                vis_point.y = y_centre + vis_y * dir_y[i];
                vis_point.z = z_centre;
  
                Eigen::Vector3i point_RGB = rrb.getRGBFromDistance(vis_point.x,
                     vis_point.y, vis_point.z);

                vis_point.r = point_RGB(0);
                vis_point.g = point_RGB(1);
                vis_point.b = point_RGB(2);

                cloud_dis_field->points.push_back(vis_point);
            }
        }
    }

    sensor_msgs::PointCloud2 cloud_env_out, cloud_dis_field_out;
    pcl::toROSMsg(*cloud_env, cloud_env_out);
    pcl::toROSMsg(*cloud_dis_field, cloud_dis_field_out);

    cloud_env_out.header.stamp = ros::Time::now();
    cloud_env_out.header.frame_id = "world";
    vis_cloud_env_pub.publish(cloud_env_out);

    cloud_dis_field_out.header.stamp = ros::Time::now();
    cloud_dis_field_out.header.frame_id = "world";
    vis_cloud_dis_field_pub.publish(cloud_dis_field_out);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_planning_pva");
    ros::NodeHandle nh;

    // State parameters initiate
    p_goal << 40.0, 0.0, 1.5;
    p0 << 0.0, 0.0, 1.5;
    v0 << 0.0, 0.0, 0.0;
    a0 << 0.0, 0.0, 0.0;

    p_store(0) = 0.0;
    p_store(1) = 0.0;
    p_store(2) = flight_altitude;

    // Fov sample parameters
    Fov_half << 35, 20;
    Angle_h << -90, -70, -50, -30, -20, -10, 0, 10, 20, 30, 50, 70, 90;
    // Angle_v << -40, -20, -10, 0, 10, 20, 40;
    Angle_v << 0;

    Angle_h = Angle_h * M_PI / 180.0;
    Angle_v = Angle_v * M_PI / 180.0;
    Fov_half = Fov_half * M_PI / 180.0;

    F_cost = Eigen::MatrixXd::Zero(ANGLE_H_NUM, ANGLE_V_NUM);
    for(int i = 0; i < ANGLE_H_NUM; i++)
    {
        for(int j = 0; j < ANGLE_V_NUM; j++)
        {
            if(fabs(Angle_h(i)) < Fov_half(0) && fabs(Angle_v(j)) < Fov_half(1)) {
                continue;
            } else
            {
                double delt_h_angle = std::min(fabs(Angle_h(i)-Fov_half(0)), fabs(Angle_h(i)+Fov_half(0)));
                double delt_v_angle = std::min(fabs(Angle_v(j)-Fov_half(1)), fabs(Angle_v(j)+Fov_half(1)));
                F_cost(i,j) = (pp.kk_h*delt_h_angle + pp.kk_v*delt_v_angle)/(270/180*M_PI); // % vertical max error + horizontal max error = 270¡ã
            }
        }
    }

    table->csv2pva_table("/home/ubuntu/test1.csv");
    table_initialized = true;

    // ringbuffer cloud2
    cloud2_pub = nh.advertise<sensor_msgs::PointCloud2>("ring_buffer/cloud_ob", 1, true);
    center_pub = nh.advertise<geometry_msgs::PointStamped>("ring_buffer/center",1,true) ;
    current_marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    
    // traj_point_pub = nh.advertise<geometry_msgs::Pose>("ring_buffer/desire_point", 5, true); // add on 9 Mar
    traj_point_pub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("/pva_setpoint", 5, true); // add on 12 Aug

    vis_cloud_env_pub = nh.advertise<sensor_msgs::PointCloud2>("/ring_buffer/vis_cloud_env", 1, true);
    vis_cloud_dis_field_pub = nh.advertise<sensor_msgs::PointCloud2>("/ring_buffer/vis_cloud_dis_field", 1, true);

    p_goal_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/p_goal_pose", 1, true);

    // Only for checking acceleration
    max_accel_pub = nh.advertise<geometry_msgs::Vector3>("ring_buffer/max_accel", 1, true);

    control_p_idx_pub = nh.advertise<std_msgs::Int16>("ring_buffer/control_point_idx", 1, true);

    ros::Subscriber odom_isolate_sub = nh.subscribe("/firefly/ground_truth/odometry", 1, odomCallback);
    ros::Subscriber imu_sub = nh.subscribe("/firefly/ground_truth/imu", 1, imuCallback);

    ros::Subscriber p_goal_sub = nh.subscribe("/keyboard/twist", 1, pGoalCallback);

    ros::Subscriber vis_cloud_sub = nh.subscribe("/ring_buffer/cloud_ob", 1, visCloudCallback);

    // message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/firefly/ground_truth/odometry", 2);
    // message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/firefly/vi_sensor/camera_depth/depth/points", 1);
    // // message_filters::Subscriber<sensor_msgs::Imu> imu_sub(nh, "/firefly/ground_truth/imu", 2);

    // TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::PointCloud2> sync(odom_sub, pcl_sub, 10);       
    // sync.registerCallback(boost::bind(&odomCloudCallback, _1, _2));   

    ros::Subscriber odom_pcl_sub = nh.subscribe("/firefly/ground_truth/odometry", 2, odomPclCallback);
    ros::Subscriber cloud_sub = nh.subscribe("/firefly/vi_sensor/camera_depth/depth/points", 1, cloudCallback);

    // TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::Imu> sync2(odom_sub, imu_sub, 10);       
    // sync2.registerCallback(boost::bind(&stateCallBack, _1, _2));              

    // timer for publish ringbuffer as pointcloud
    ros::Timer timer1 = nh.createTimer(ros::Duration(0.2), timerCallback); // RATE 5 Hz to publish

    // timer for trajectory generation
    ros::Timer timer2 = nh.createTimer(ros::Duration(cal_duration), trajectoryCallback); 

    std::cout << "Start mapping!" << std::endl;

    // ros::spin();
    ros::AsyncSpinner spinner(4); // Use 4 threads
    spinner.start();
    ros::waitForShutdown();

    return 0;
}