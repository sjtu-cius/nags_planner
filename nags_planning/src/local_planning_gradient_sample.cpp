#include <ewok/ed_ckecking_ring_buffer.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Point32.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <mavros_msgs/State.h> 
#include <nav_msgs/Odometry.h>
#include <control_msgs/JointControllerState.h>
#include <std_msgs/Float64.h>
#include <queue>
#include <cstdint>
#include <trajectory_msgs/JointTrajectoryPoint.h>

using namespace message_filters;
using namespace std;

#define PIx2 6.28318
#define PI 3.14159
#define PI_2 1.5708

/**** Parameters to tune, some initialization needs to be changed in main function ****/
const double resolution = 0.13;
const double trunc_distance = 1.0;
static const int POW = 6;
static const int N = (1 << POW);
const float SEND_DURATION = 0.0250f; //40Hz (10HZ at least)
ewok::EuclideanDistanceCheckingRingBuffer<POW> rrb(resolution, trunc_distance); //Distance truncation threshold

const int LOOKING_PIECES_SIZE = 16; // Should be even, Resolution: 22.5 degrees one piece. MID=7. Yaw=0. Larger: 7 - 15 :  0 to Pi;  Smaller: 7-0 : 0 to (nearly)-Pi;
float CAMERA_H_FOV = 62.f;  //degrees, in parameter list
const float HEAD_BUFFER_HIT_INCREASE = 0.4f;
const float HEAD_BUFFER_MISS_DECREASE_STANDARD_V = 0.05f; // Miss add value when velocity is 1m/s
const float HEAD_BUFFER_MISS_DECREASE_MIN = 0.03f;

float HEIGHT_LIMIT, XY_LIMIT, MAX_V_XY, MAX_V_Z_UP, MAX_V_Z_DOWN, MAX_A;

float PLAN_INIT_STATE_CHANGE_THRESHOLD = 0.1;

struct  Path_Planning_Parameters
{
    double d_ref = 1.6;
    double k1_xy = 5;
    double k1_z = 3;
    double k2_xy = 1;
    double k2_z = 1.5;
    double k3_xy = 0.5;
    double k3_z = 0.5;
    double v_max_at_goal_ori = 0.3; //% m/s, just reference  5.0 originally
    double v_scale_min = 0.3;
    double collision_threshold_static = 0.6;
}pp;


struct  Head_Planning_Parameters
{
   double k_current_v = 0.6;
   double k_planned_direction = 0.2;
   double k_v_fluctuation = 0.3;
   double k_common_update = 0.1;
}hp;

const int point_num_pub = 5; // For the visualization of current planned trajectory
/*** End of Parameters ***/

/** Basic global variables **/
ros::Publisher current_marker_pub;
ros::Publisher cloud2_pub, cloud_edf_pub, goal_pub;
ros::Publisher map_center_pub;

ros::Publisher head_cmd_pub; // add on 9 Sept.
ros::Publisher motor_pub;
ros::Publisher pva_pub;  //add on 20 Aug. 2020

ros::Publisher sim_trajectory_pub; // add on 7 Jan. 2020
double x_centre, y_centre, z_centre;

bool if_in_simulation = false;
bool offboard_ready = false;
bool state_locked = false;
bool state_updating = false;
bool safe_trajectory_avaliable = true;
bool if_plan_vertical_path = true;
bool if_sim_lee_position_controller = false;
bool if_flyback_enable = true;

std::queue<double> pose_timestamp_queue;
std::queue<Eigen::Quaternionf> att_queue;
std::queue<Eigen::Vector3d> pos_queue;

std::string TABLE_PATH;

/****** Global variables for path planning ******/
ros::Time _data_input_time;

bool initialized = false;

Eigen::Vector3d p_goal;
Eigen::Vector3d p_goal_raw;
Eigen::Vector3d p0;
Eigen::Vector3d v0;
Eigen::Vector3d a0;
Eigen::Quaternionf quad(1.0, 0.0, 0.0, 0.0);
Eigen::Vector3d init_p;
double collected_time;
double yaw0;
double yaw0_rate;
double max_sample_delt_angle = 1.0;

/*** Global variables for rotating head ***/
Eigen::VectorXf _direction_update_buffer; // Range [0, 1]
float heading_resolution;
int valid_piece_num_one_side;
int mid_seq_num; 

double motor_yaw = 0.0;
double motor_yaw_rate = 0.0;
double head_max_yaw_delt = 0.5;
double motor_velocity_set = 50.0;

double init_head_yaw = 0.0;


/** Publishers for cost visualization **/
bool if_publish_panel_arrays = true;
ros::Publisher cost_head_update_degree_pub;
ros::Publisher cost_head_velocity_pub;
ros::Publisher cost_head_direction_pub;
ros::Publisher cost_head_objects_pub;
ros::Publisher cost_head_fluctuation_pub;
ros::Publisher cost_head_final_pub;

/** Declaration of functions**/

void sendMotorCommands(double yaw);

void setParameters();

void getParameterList(ros::NodeHandle nh);

void setPointSend(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Vector3d a);

/**********************FOR PVA TABLE***************************/
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

        return this->table[idx];
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
};

pva_table *trajectory_time_table = (pva_table*)malloc(sizeof(pva_table));

/***************************************************/
double getHeadingYawFromSeq(int seq)
{
    if(seq >= LOOKING_PIECES_SIZE || seq < 0){
        ROS_ERROR("Seq for yaw buffer of the head out of range.");
        return 0.f;
    }
    else{
        return (seq - mid_seq_num) * heading_resolution;
    }
}

int getHeadingSeq(double direction)
{
    if(direction > PI){
        direction -= PIx2;
    }
    else if(direction < -PI){
        direction += PIx2;
    } 

    if(direction > 0){  // To sovle the truncation problem
        direction += heading_resolution / 2.f;
    }
    else{
        direction -= heading_resolution / 2.f;  
    }
    /// Caution!!!!! the rotating range of the camera can only be within (-PI-heading_resolution, PI+heading_resolution), chg
    int heading_seq =  (int)(direction / heading_resolution) + mid_seq_num;
    if(heading_seq == LOOKING_PIECES_SIZE) heading_seq = 0;
    else if(heading_seq < 0) heading_seq = LOOKING_PIECES_SIZE - 1;

    return heading_seq;
}

void correctPieceSeq(int &seq)
{
    if(seq >= LOOKING_PIECES_SIZE){   // to form a ring
        seq = seq - LOOKING_PIECES_SIZE;
    }
    else if(seq < 0){
        seq = seq + LOOKING_PIECES_SIZE;
    }
}

void addHitOnePiece(int seq)
{
    _direction_update_buffer(seq)+HEAD_BUFFER_HIT_INCREASE<=1.f ? _direction_update_buffer(seq)+=(float)HEAD_BUFFER_HIT_INCREASE : _direction_update_buffer(seq)=1.f;
}

void addMissOnePiece(int seq)
{
    float delt_miss = HEAD_BUFFER_MISS_DECREASE_MIN;
    float delt_miss_by_velocity = HEAD_BUFFER_MISS_DECREASE_STANDARD_V * std::max(fabs(v0(0)), fabs(v0(1)));
    if(delt_miss_by_velocity > delt_miss){
        delt_miss = delt_miss_by_velocity;
    }

    _direction_update_buffer(seq)-delt_miss>=0.f ? _direction_update_buffer(seq)-=delt_miss : _direction_update_buffer(seq)=0.f;
}

void updateHeadBuffer(const int &heading_direction_seq)
{
    Eigen::VectorXi update_flag_buf = Eigen::VectorXi::Zero(LOOKING_PIECES_SIZE);

    //Add hit for the direction in FOV
    addHitOnePiece(heading_direction_seq);
    update_flag_buf(heading_direction_seq) = 1; 

    for(int i=1; i<=valid_piece_num_one_side; i++){
        int seq_this_1 = heading_direction_seq + i;
        int seq_this_2 = heading_direction_seq - i;
        
        correctPieceSeq(seq_this_1);
        correctPieceSeq(seq_this_2);

        addHitOnePiece(seq_this_1);
        addHitOnePiece(seq_this_2);

        update_flag_buf(seq_this_1) = 1;
        update_flag_buf(seq_this_2) = 1;
    }
    //Add miss for the rest
    for(int j=0; j<LOOKING_PIECES_SIZE; j++){
        if(update_flag_buf(j) != 1){
            addMissOnePiece(j);
        } 
    }
}


bool motor_initialized = false;
// this callback use input cloud to update ring buffer, and update odometry of UAV
void cloudCallback(const control_msgs::JointControllerStateConstPtr &motor_msg, const sensor_msgs::PointCloud2ConstPtr& cloud)
{
    /*** For motor ***/
    static bool init_time = true;

    if(init_time)
    {
        init_head_yaw = motor_msg->process_value;
        init_time = false;
        motor_initialized = true;
        ROS_INFO("Head Init Yaw in motor coordinate=%f", init_head_yaw);
    }
    else
    {
        if(!if_in_simulation){  //real world
            motor_yaw = -motor_msg->process_value + init_head_yaw; // + PI_2?? //start with zero, original z for motor is down. now turn to ENU coordinate. Head forward is PI/2 ???????????
        }else{
            motor_yaw = motor_msg->process_value - init_head_yaw;
        }
        motor_yaw_rate = 0.0;

        while(motor_yaw > PI){
            motor_yaw -= PIx2;
        }
        while(motor_yaw < -PI){
            motor_yaw += PIx2;
        }

        geometry_msgs::Point32 real_motor_pv;
        real_motor_pv.x = motor_yaw;
        real_motor_pv.y = motor_yaw_rate;
        real_motor_pv.z = 0;

        motor_pub.publish(real_motor_pv);
    }

    /*** For cloud ***/
    _data_input_time = ros::Time::now();

    Eigen::Quaternionf quad_sychronized = quad;  //initialze in case no point in the queue satisfy
    Eigen::Vector3d current_position = p0;

    while(!pose_timestamp_queue.empty()){   //Sychronize pose by queue
        double time_stamp_pose = pose_timestamp_queue.front();
        if(time_stamp_pose >= cloud->header.stamp.toSec()){
            quad_sychronized = att_queue.front();
            current_position = pos_queue.front();
            ROS_INFO_THROTTLE(3.0, "cloud mismatch time = %lf", cloud->header.stamp.toSec() - time_stamp_pose);
            break;
        }
        pose_timestamp_queue.pop();
        att_queue.pop();
        pos_queue.pop();
    }

    // Add the rotation of head
    Eigen::Quaternionf axis_motor;
    axis_motor.w() = cos(motor_yaw/2.0);
    axis_motor.x() = 0;
    axis_motor.y() = 0;
    axis_motor.z() = sin(motor_yaw/2.0);
    Eigen::Quaternionf quad_rotate = quad_sychronized * axis_motor;

    // create transform matrix
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) = Eigen::Matrix3f(quad_rotate);
    transform(0, 3) = p0(0);
    transform(1, 3) = p0(1);
    transform(2, 3) = p0(2);

    // convert cloud to pcl form
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud, *cloud_in);

    // down-sample for all
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_in);
    static float res = resolution;
    sor.setLeafSize(res, res, res);
    sor.filter(*cloud_filtered);

    // transform to world NWU frame
    Eigen::Matrix4f t_c_b = Eigen::Matrix4f::Zero();
    t_c_b(0, 2) = 1;
    t_c_b(1, 0) = -1;
    t_c_b(2, 1) = -1;
    t_c_b(3, 3) = 1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud_filtered, *cloud_1, t_c_b);
    pcl::transformPointCloud(*cloud_1, *cloud_2, transform);

    // compute ewol pointcloud and origin
    Eigen::Vector3f origin = (transform * Eigen::Vector4f(0, 0, 0, 1)).head<3>(); //  position (x,y,z)
    ewok::EuclideanDistanceCheckingRingBuffer<POW>::PointCloud cloud_ew;
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > points = cloud_2->points; //  cloud_2->points;

    x_centre = p0(0);
    y_centre = p0(1);
    z_centre = p0(2);

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
        ROS_WARN("Initialized!!");
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

    double preprocess = ros::Time::now().toSec() - _data_input_time.toSec();

    // insert point cloud to ringbuffer
    rrb.insertPointCloud(cloud_ew, origin);

    double insert_t = ros::Time::now().toSec() - _data_input_time.toSec();


    // Calculate distance field considering newly imported points (dynamic points)
    //rrb.updateDistanceDynamic(cloud_ew, origin);
    rrb.updateDistance();

    double elp = ros::Time::now().toSec() - _data_input_time.toSec();
    ROS_INFO_STREAM_THROTTLE(3.0, "Map update time = " << elp << " s");

    /* Update buffer for rolling head */
    double head_update_start_time = ros::Time::now().toSec();
    int heading_direction_seq = getHeadingSeq(motor_yaw);  //current heading direction
    updateHeadBuffer(heading_direction_seq);
    double head_update_time = ros::Time::now().toSec() - head_update_start_time;
    ROS_INFO_THROTTLE(3.0, "SUD update time = %lf", head_update_time);
}

void cloudPubCallback(const ros::TimerEvent& e)
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

    //Publish center position
    geometry_msgs::PointStamped center_p;
    center_p.header.stamp = ros::Time::now();
    center_p.header.frame_id = "world";
    center_p.point.x = center(0);
    center_p.point.y = center(1);
    center_p.point.z = center(2);
    map_center_pub.publish(center_p);
    //ROS_INFO("Cloud published!");

    /*EDF showing*/
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_edf_field(new pcl::PointCloud<pcl::PointXYZRGB>());
    static double boundary = pow(2.0, POW) * resolution / 2; // Here the boundary is 6.4m

    for (double vis_x = resolution / 2; vis_x < boundary; vis_x += resolution) {
        for (double vis_y = resolution / 2; vis_y < boundary; vis_y += resolution) { //遍历整个空间填充distance颜色
            // Skip the truncation area
            if (vis_x <= 0.2 && vis_y <= 0.2) {
                continue;
            }
            pcl::PointXYZRGB vis_point;
            int dir_x[4] = {1, 1, -1, -1};
            int dir_y[4] = {1, -1, 1, -1};

            for (int i = 0; i < 4; ++i) {
                vis_point.x = x_centre + vis_x * dir_x[i];
                vis_point.y = y_centre + vis_y * dir_y[i];
                vis_point.z = z_centre;

                Eigen::Vector3i point_RGB = rrb.get_rgb_edf(vis_point.x, vis_point.y, vis_point.z);

                vis_point.r = point_RGB(0);
                vis_point.g = point_RGB(1);
                vis_point.b = point_RGB(2);

                cloud_edf_field->points.push_back(vis_point);
            }
        }
    }
    // Publish edf as point cloud
    sensor_msgs::PointCloud2 edf_ros_cloud;
    pcl::toROSMsg(*cloud_edf_field, edf_ros_cloud);
    edf_ros_cloud.header.stamp = ros::Time::now();
    edf_ros_cloud.header.frame_id = "world";
    cloud_edf_pub.publish(edf_ros_cloud);
}


/**
 * Generate a path with initial state and final state. Consider the kinodynamics by a table.
 * Criterion for a feasible path:
 * 1. The acceleration and velocity is with in allowed range. (Defined when generating the table.)
 * 2. The minimum distance between the path and obstacles is larger than dist min or
 *    the path is leading to a way far from obstacles.
 * */
int generate_primitive_with_table_and_check_feasibility(const Eigen::Vector3d &p0, const Eigen::Vector3d &v0_ori, const Eigen::Vector3d &a0_ori, const double yaw0, const double theta_h,
                                                         const double theta_v, const Eigen::Vector3d &goal, const double path_length, const double v_final, const double delt_t, const double allow_min_dist,
                                                         Eigen::MatrixXd &p, Eigen::MatrixXd &v, Eigen::MatrixXd &a, Eigen::VectorXd &t, double &min_dist)
{
    double delt_x = path_length*cos(theta_v)*cos(theta_h+yaw0);
    double delt_y = path_length*cos(theta_v)*sin(theta_h+yaw0);
    double delt_z = path_length*sin(theta_v);

    Eigen::Vector3d pf;
    pf << p0(0)+delt_x, p0(1)+delt_y, p0(2)+delt_z;

    Eigen::Vector3d l = goal - pf;
    Eigen::Vector3d vf = (v_final / l.norm()) * l;
    vf(2) = 0; // % Note: 0 maybe better, for the p curve wont go down to meet the vf

    Eigen::Vector3d af = Eigen::Vector3d::Zero();

    Eigen::Vector3d v0 = v0_ori;
    Eigen::Vector3d a0 = a0_ori;

    for(int i=0; i<3; i++){
        if(fabs(v0(i)) > (MAX_V_XY - 0.1)) {
            v0(i) = (MAX_V_XY - 0.1) * v0(i) / fabs(v0(i));
            ROS_WARN_THROTTLE(1, "v_limit in local planning is large than the values in the loaded table!  v0 max=%f", v0.cwiseAbs().maxCoeff());
        }
        if(fabs(a0(i)) > (MAX_A-0.1)) {
            a0(i) = (MAX_A-0.1) * a0(i) / fabs(a0(i));
            ROS_WARN_THROTTLE(1, "a_limit in local planning is large than the values in the loaded table!  a0 max=%f", a0.cwiseAbs().maxCoeff());
        }
    }

    double T1, T2, T3, T;
    T1 = trajectory_time_table->query_pva_table(delt_x, v0(0), vf(0), a0(0));
    T2 = trajectory_time_table->query_pva_table(delt_y, v0(1), vf(1), a0(1));
    T3 = trajectory_time_table->query_pva_table(delt_z, v0(2), vf(2), a0(2));

    if (T1 == -1 || T2 == -1 || T3 == -1) {
        min_dist = 0;  ///No feasible path. Treat as distance is zero.
        return -1;
    }


    T = T1 > T2 ? T1 : T2;
    T = T > T3 ? T : T3;
    T = T < 0.5 ? 0.5 : T;

    int times = T / delt_t;

    p = Eigen::MatrixXd::Zero(times, 3);
    v = Eigen::MatrixXd::Zero(times, 3);
    a = Eigen::MatrixXd::Zero(times, 3);
    t = Eigen::VectorXd::Zero(times);

    // calculate optimal jerk control points
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
        }
    }

    const int Num = times / 2;  //Check every two points
    Eigen::Vector3f *sim_traj = new Eigen::Vector3f[Num];
    for (int i = 0; i < Num; i++) {
        sim_traj[i](0) = (float)p.row(i*2)(0);
        sim_traj[i](1) = (float)p.row(i*2)(1);
        sim_traj[i](2) = (float)p.row(i*2)(2);
    }

    float min_distance;
    bool if_far_from_obstacles;
    /// Use Function "collision_checking_with_fence_min_dist" if a virtual fence is considered
    //if_far_from_obstacles = rrb.collision_checking_with_fence_min_dist(sim_traj, Num, min_distance, init_p, HEIGHT_LIMIT, XY_LIMIT);

    if_far_from_obstacles = rrb.collision_checking_no_fence_min_dist(sim_traj, Num, min_distance);

    min_dist = (double)min_distance;

    if(if_far_from_obstacles){  ///Safe path getting away from obstacles
        return 1;
    }

    if(min_dist > allow_min_dist){  ///Safe path far from obstacles
        return 2;
    }

    return -2;
}


/*****************
 * This function do sample and collision checking.
 * Sample range is [theta_h_goal-theta_h_range_delt_max, theta_h_goal+theta_h_range_delt_max] horizontally
 * and [-theta_v_range_delt_max, theta_v_range_delt_max] vertically.
 * @Param: lamda_goal and lamda_last are coefficients used to give an optimal initial sample direction.
 * @Param: delt_theta_max_h and delt_theta_max_v are the maximum angle differece between two samples.
 * @Param: max_iteration_times defines the maximum iteration times to find a feasible path.
 *****************/
int feasible_path_generate(double path_length, double dist_threshold, double theta_h_range_delt_max, double theta_v_range_delt_max,
                            double lamda_goal, double lamda_last, double lamda_head, double delt_t, const Eigen::Vector3d &p0, const Eigen::Vector3d &v0,
                            const Eigen::Vector3d &a0, double yaw0, const Eigen::Vector3d& p_goal, double v_final,
                            Eigen::MatrixXd &p, Eigen::MatrixXd &v, Eigen::MatrixXd &a, Eigen::VectorXd &t, double &theta_h_chosen, //result
                            double delt_theta_max_h = 0.785f, double delt_theta_max_v = 0.4f, int max_iteration_times = 10)
{
    static double theta_h_last, theta_v_last;
    Eigen::Vector3d delt_p = p_goal - p0;
    double theta_h_goal, theta_v_goal;
    theta_h_goal = atan2(delt_p(1), delt_p(0));
    theta_v_goal = atan2(delt_p(2), sqrt(delt_p(0) * delt_p(0) + delt_p(1) * delt_p(1)));

    static bool init_flag = true;
    if(init_flag){
        theta_h_last = theta_h_goal;
        theta_v_last = theta_v_goal;
        init_flag = false;
    }

    double theta_star_h = (lamda_goal*theta_h_goal + lamda_last*theta_h_last + lamda_head*motor_yaw) / (lamda_goal+lamda_last+lamda_head);
    double theta_star_v = (lamda_goal*theta_v_goal + lamda_last*theta_v_last + lamda_head*motor_yaw) / (lamda_goal+lamda_last+lamda_head);

    double theta_h_range_delt_max_positive = theta_h_range_delt_max;// + (theta_h_goal - theta_star_h);  /// delete (theta_h_goal - theta_star_h) to sample around theta_star_h
    double theta_h_range_delt_max_negative = -theta_h_range_delt_max;// + (theta_h_goal - theta_star_h);

    int feasible_flag = 0;
    double banned_theta_h_delt_positive = 0.f;
    double banned_theta_h_delt_negative = -0.f;
    double banned_theta_h_delt_positive_temp = 0.f;
    double banned_theta_h_delt_negative_temp = -0.f;

    double theta_h_1, theta_v_1, delt_theta, d0, d1;
    int same_d_counter = 0;
    int fail_reason_dist_counter = 0, fail_reason_path_generate_counter = 0;

    for(int iter_times = 0; iter_times < max_iteration_times; iter_times++){
        banned_theta_h_delt_positive = banned_theta_h_delt_positive_temp;
        banned_theta_h_delt_negative = banned_theta_h_delt_negative_temp;

        if(banned_theta_h_delt_positive > theta_h_range_delt_max_positive && banned_theta_h_delt_negative < theta_h_range_delt_max_negative){
            break;
        }
        ///Find start theta
        if(fabs(banned_theta_h_delt_positive) <= fabs(banned_theta_h_delt_negative)){
            theta_h_1 = theta_star_h + banned_theta_h_delt_positive;
            theta_v_1 = theta_star_v;
            delt_theta = 0.4f;
        }else{
            theta_h_1 = theta_star_h + banned_theta_h_delt_negative;
            theta_v_1 = theta_star_v;
            delt_theta = -0.4f;
        }

        feasible_flag = generate_primitive_with_table_and_check_feasibility(p0, v0, a0, yaw0, theta_h_1, theta_v_1,
                                                 p_goal, path_length, v_final,delt_t, dist_threshold, p, v, a, t, d0);
        double d0_h_first = d0;
        double theta_h_1_first = theta_h_1;

        if(feasible_flag > 0){
            break;
        }else{
            if(feasible_flag == -1){
                fail_reason_path_generate_counter += 1;
            } else{
                fail_reason_dist_counter += 1;
            }
        }

        same_d_counter = 0;
        /// Horizontally check
        for(int iter_unbanned_area=0; iter_unbanned_area<4; iter_unbanned_area++){
            double last_theta_h_1 = theta_h_1;
            theta_h_1 = theta_h_1 + delt_theta;

            if(theta_h_1 > theta_star_h+theta_h_range_delt_max_positive || theta_h_1 < theta_star_h+theta_h_range_delt_max_negative){
                /// theta_h_1 out of range
                theta_h_1 = last_theta_h_1;
                break;
            }
            /// Save for banned area
            double delt_to_star_h = theta_h_1 - theta_star_h;
            if(delt_to_star_h > banned_theta_h_delt_positive_temp){
                banned_theta_h_delt_positive_temp = delt_to_star_h;
            }else if(delt_to_star_h < banned_theta_h_delt_negative_temp){
                banned_theta_h_delt_negative_temp = delt_to_star_h;
            }
            /// Check feasibility
            feasible_flag = generate_primitive_with_table_and_check_feasibility(p0, v0, a0, yaw0, theta_h_1, theta_v_1,
                                                                                p_goal, path_length, v_final,delt_t, dist_threshold, p, v, a, t, d1);
            if(feasible_flag > 0){
                break;
            }else{
                if(feasible_flag == -1){
                    fail_reason_path_generate_counter += 1;
                } else{
                    fail_reason_dist_counter += 1;
                }
            }

            /// Not satisfied. Go on iteration
            double diff_d = (d1 - d0) / delt_theta;
            if(fabs(diff_d) > 0.001){
                delt_theta = (dist_threshold - d1) / diff_d;
                delt_theta = std::min(delt_theta, delt_theta_max_h);
                delt_theta = std::max(delt_theta, -delt_theta_max_h);
                if(fabs(delt_theta) < 0.05){
                    if(delt_theta > 0) delt_theta = 0.05;
                    else delt_theta = -0.05;
                }
                d0 = d1;
                same_d_counter = 0;
            }else{
                if(d1>0){
                    if(same_d_counter < 3){
                        d0 = d1;
                        same_d_counter += 1;
                        /// Delt theta keeps last value
                    }else{
                        /// d keeps to be the same value. Set a banned area later.
                        break;
                    }
                }else{
                    /// d is zero. Set a banned area later.
                    break;
                }
            }
        }
        if(feasible_flag > 0){break;}

        /// Vertically check
        same_d_counter = 0;
        d1 = d0_h_first;
        theta_h_1 = theta_h_1_first;

        for(int j=0; j<2; j++){
            if(!if_plan_vertical_path){
                break;
            }

            theta_v_1 = theta_star_v;
            d0 = d1;
            delt_theta = 0.52 * std::pow(-1, j);  //30 degrees
            for(int i=0;i<4;i++){
                theta_v_1 = theta_v_1 + delt_theta;
                if(theta_v_1 > theta_star_v + theta_v_range_delt_max || theta_v_1 < theta_star_v - theta_v_range_delt_max){
                    break;
                }
                feasible_flag = generate_primitive_with_table_and_check_feasibility(p0, v0, a0, yaw0, theta_h_1, theta_v_1,
                                                                                    p_goal, path_length, v_final,delt_t, dist_threshold, p, v, a, t, d1);

                if(feasible_flag > 0){
                    ROS_WARN("Found vertically feasible path, p0(0)=%f, d1=%f, theta_h_1=%f, theta_v_1=%f", p0(0), d1, theta_h_1, theta_v_1);
                    break;
                }else{
                    if(feasible_flag == -1){
                        fail_reason_path_generate_counter += 1;
                    } else{
                        fail_reason_dist_counter += 1;
                    }
                }

                double diff_d = (d1 - d0) / delt_theta;
                if(fabs(diff_d) > 0.001){
                    delt_theta = (dist_threshold - d1) / diff_d;
                    delt_theta = std::min(delt_theta, delt_theta_max_v);
                    delt_theta = std::max(delt_theta, -delt_theta_max_v);
                    if(fabs(delt_theta) < 0.05){
                        if(delt_theta > 0) delt_theta = 0.05;
                        else delt_theta = -0.05;
                    }
                    d0 = d1;
                }else{
                    d0 = d1;
                }
            }
            if(feasible_flag > 0){break;}
        }
    }

    if(feasible_flag > 0){
        theta_h_chosen = theta_h_1;  //theta_h_chosen is used in head direction planning
        while(theta_h_chosen > PI){  //Transform to [-pi, pi]
            theta_h_chosen -= PIx2;
        }
        while(theta_h_chosen < -PI){
            theta_h_chosen += PIx2;
        }
        theta_h_last = theta_h_chosen;
        theta_v_last = theta_v_1;

        ROS_INFO_THROTTLE(0.5, "Found feasible path. Feasible Code: %d, Spent times: Path infeasibility: %d, Close to obstacles: %d", feasible_flag, fail_reason_path_generate_counter, fail_reason_dist_counter);
    }else{
        theta_h_chosen = theta_h_last;
        ROS_WARN_THROTTLE(0.5, "Found no feasible path. Reason: Path infeasibility: %d, Close to obstacles: %d", fail_reason_path_generate_counter, fail_reason_dist_counter);
    }

    /*** Code to calculate average sample times ***/
    static long long total_times = 0;
    static long long calculate_times = 0;
    static long long total_failure_times_path_generate = 0;
    static long long total_fail_reason_dist_counter = 0;
    static double flight_start_time = ros::Time::now().toSec();

    total_fail_reason_dist_counter += fail_reason_dist_counter;
    total_failure_times_path_generate += fail_reason_path_generate_counter;
    calculate_times += 1;

    if(feasible_flag > 0){
        total_times = total_times + fail_reason_dist_counter + fail_reason_path_generate_counter + 1;
    }else{
        total_times = total_times + fail_reason_dist_counter + fail_reason_path_generate_counter;
    }
    double avg_sample_times, avg_path_generate_failure_times, avg_dist_failure_times;
    avg_sample_times = (double)total_times / calculate_times;
    avg_path_generate_failure_times = (double)total_failure_times_path_generate / calculate_times;
    avg_dist_failure_times = (double)total_fail_reason_dist_counter / calculate_times;

    ROS_INFO_THROTTLE(0.5, "Current time=%f s, avg_sample_times=%f, avg_path_generate_failure_times=%f, avg_dist_failure_times=%f",
            ros::Time::now().toSec()-flight_start_time,avg_sample_times, avg_path_generate_failure_times, avg_dist_failure_times);

    /*** END ***/

    return feasible_flag;
}

/* Publish markers to show the path in rviz */
void marker_publish(Eigen::MatrixXd &Points)
{
    visualization_msgs::Marker points, goal_strip;
    goal_strip.header.frame_id = points.header.frame_id = "world";
    goal_strip.header.stamp = points.header.stamp = ros::Time::now();
    goal_strip.action = points.action = visualization_msgs::Marker::ADD;
    goal_strip.ns=points.ns = "lines_and_points";
    points.id = 1;
    goal_strip.id = 0;
    goal_strip.type = visualization_msgs::Marker::LINE_STRIP;
    points.type = visualization_msgs::Marker::POINTS;


    // Line width
    points.scale.x = 0.1;
    goal_strip.scale.x = 0.05;

    // points are green
    points.color.g = 1.0;
    points.color.a = 1.0;
    points.lifetime = ros::Duration(0);

    goal_strip.color.g = 0.8;
    goal_strip.color.b = 0.8;
    goal_strip.color.r = 1.0;
    goal_strip.color.a = 1.0;
    goal_strip.lifetime = ros::Duration(0);

    int point_num = Points.rows();

    /** Add current position **/
    geometry_msgs::Point p;
    p.x = p0(0);
    p.y = p0(1);
    p.z = p0(2);
    points.points.push_back(p);
    goal_strip.points.push_back(p);

    /** Add planned path **/
    for(int i=0; i<point_num; i++)
    {
        geometry_msgs::Point p;
        p.x = Points(i, 0);
        p.y = Points(i, 1);
        p.z = Points(i, 2);

        points.points.push_back(p);
    }

    current_marker_pub.publish(points);

    geometry_msgs::Point p2;
    p2.x = p0(0);
    p2.y = p0(1);
    p2.z = p0(2);
    goal_strip.points.push_back(p2);
    p2.x = p_goal(0);
    p2.y = p_goal(1);
    p2.z = p_goal(2);
    goal_strip.points.push_back(p2);
    current_marker_pub.publish(goal_strip);
}


/** This is the function to generate the collision-free path, which is trigered by a timer defined in main function. **/
void trajectoryCallback(const ros::TimerEvent& e) {

    double  algorithm_start_time = ros::Time::now().toSec();

    /** Generate trajectory for uav first***/
    int collision_check_pass_flag = 0;
    double theta_h_chosen;

    static Eigen::Vector3d last_sended_p = p0;
    static Eigen::Vector3d last_sended_v = v0;
    static Eigen::Vector3d last_sended_a = a0;

    if(!state_updating)
    {
        state_locked = true;

        /** Moition primitives **/
        Eigen::Vector3d delt_p = p_goal - p0;
        double v_scale = std::max(delt_p.dot(v0)/v0.norm()/delt_p.norm(), pp.v_scale_min);
        // If v0.norm() = 0, v_scale = nan
        v_scale = (v0.norm() == 0) ? pp.v_scale_min : v_scale;
        double v_final = pp.v_max_at_goal_ori * v_scale;

        ROS_INFO_THROTTLE(1, "v_final=%f", v_final);

        Eigen::MatrixXd p;
        Eigen::MatrixXd v;
        Eigen::MatrixXd a;
        Eigen::VectorXd t;

        if((last_sended_p - p0).squaredNorm() < PLAN_INIT_STATE_CHANGE_THRESHOLD * PLAN_INIT_STATE_CHANGE_THRESHOLD){
            collision_check_pass_flag = feasible_path_generate(pp.d_ref, pp.collision_threshold_static, max_sample_delt_angle, 1.2,
                                                               pp.k1_xy, pp.k2_xy, pp.k3_xy, SEND_DURATION, last_sended_p, last_sended_v, last_sended_a,
                                                               0, p_goal, v_final, p, v, a, t, theta_h_chosen);
            ROS_INFO_THROTTLE(3.0, "### Continue to plan!");
        }else{
            collision_check_pass_flag = feasible_path_generate(pp.d_ref, pp.collision_threshold_static, max_sample_delt_angle, 1.2,
                                                               pp.k1_xy, pp.k2_xy, pp.k3_xy, SEND_DURATION, last_sended_p*0.1 + p0*0.9, last_sended_v*0.1+v0*0.9, a0,
                                                               0, p_goal, v_final, p, v, a, t, theta_h_chosen);
            ROS_WARN_THROTTLE(2.0, "### Track point too far, replan! last_sended_p=(%f, %f, %f), p0=(%f, %f, %f)", last_sended_p(0), last_sended_p(1), last_sended_p(2), p0(0), p0(1), p0(2));
        }

        if(collision_check_pass_flag > 0){
            safe_trajectory_avaliable = true;

            int send_seq = 1;
            setPointSend(p.row(send_seq), v.row(send_seq), a.row(send_seq));  ///SEND
            last_sended_p = p.row(send_seq);
            last_sended_v = v.row(send_seq);
            last_sended_a = a.row(send_seq);

            //Publish down sampled path points
            int interval_num = p.rows() / point_num_pub;
            if (interval_num > 0)
            {
                Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(point_num_pub+1, 3);
                for(int pub_i = 0; pub_i < point_num_pub; pub_i++)
                {
                    show_points.row(pub_i) = p.row(pub_i*interval_num);
                }
                show_points.row(point_num_pub) = p.row(p.rows()-1);
                marker_publish(show_points);

            }
        }else{
            ROS_WARN_THROTTLE(2.0, "No valid trajectory found! Trapped in safety mode!");
            safe_trajectory_avaliable = false;

            // Publish points of stored point to show
            Eigen::MatrixXd show_points = Eigen::MatrixXd::Zero(1, 3);
            show_points(0, 0) = p0(0);
            show_points(0, 1) = p0(1);
            show_points(0, 2) = p0(2);
            marker_publish(show_points);
        }
        state_locked = false;
    }

    double algo_time = ros::Time::now().toSec() - algorithm_start_time;
    ROS_INFO_THROTTLE(3.0, "path planning algorithm time is: %lf", algo_time);

    /** Generate trajectory for rotating head **/
    static double last_head_yaw_plan = 0.0;
    static bool first_head_control_flag = true;

    double head_planning_start_time = ros::Time::now().toSec();

    if(first_head_control_flag){
        if(motor_initialized){
            first_head_control_flag = false;
            sendMotorCommands(0.0);
        }
    }
    else{
        /// Rank by the cost given from current velocity direction, the sampled last direction, update degree and the last planned head direction
        Eigen::VectorXd cost_vector = Eigen::VectorXd::Zero(LOOKING_PIECES_SIZE);

        /// Current velocity direction(v_direction), planned velocity direction(theta_h_chosen), yaw of the head must be in the same coordinate!!
        double yaw_to_send;

        /// Larger update degree indicates the influence of this item
        double k_current_v = 0;
        static double v_direction = 0;
        if(fabs(v0(0)) > 0.05 || fabs(v0(1) > 0.05)){  // if velocity is large, use this cost from velocity direction
            // Give a discount on hp.k_current_v (max) so that when the velocity of the drone is low, the influence of the velocity direction is small.
            k_current_v =  std::max(std::min(std::max(fabs(v0(0)/MAX_V_XY), fabs(v0(1)/MAX_V_XY)), 1.0), 0.0);
            v_direction = atan2(v0(1), v0(0));
        }
        double coefficient_current_v =  k_current_v * (1.0 - _direction_update_buffer(getHeadingSeq(v_direction)));

        double coefficient_planned_dir = 0.0;
        if(collision_check_pass_flag > 0) coefficient_planned_dir = 1.0 - _direction_update_buffer(getHeadingSeq(theta_h_chosen));

        /** Visualization variables **/
        std_msgs::Float64MultiArray cost_current_velocity_array;
        std_msgs::Float64MultiArray cost_planned_direction_array;
        std_msgs::Float64MultiArray cost_head_fluctuation_array;
        std_msgs::Float64MultiArray cost_dynamic_objects_array; //useless in this file.
        std_msgs::Float64MultiArray cost_total_array;
        std_msgs::Float64MultiArray cost_update_array;
        /** End of visualization **/

        double min_head_plan_cost = 10000000.0;  //smaller cost is better
        for(int i=0; i<LOOKING_PIECES_SIZE; i++){
            double head_yaw_plan_temp = getHeadingYawFromSeq(i);

            double cost_current_velocity = hp.k_current_v * coefficient_current_v * (v_direction-head_yaw_plan_temp) * (v_direction-head_yaw_plan_temp);

            double cost_update_degree = hp.k_common_update * _direction_update_buffer(i);

            double cost_planned_direction = hp.k_planned_direction  * coefficient_planned_dir * (theta_h_chosen-head_yaw_plan_temp) * (theta_h_chosen-head_yaw_plan_temp);
            double cost_head_fluctuation = hp.k_v_fluctuation * (head_yaw_plan_temp - last_head_yaw_plan) * (head_yaw_plan_temp - last_head_yaw_plan);

            double cost_dynamic_objects = 0.0;
            static int effect_range_one_side = 3;

            double cost_total_temp = cost_current_velocity + cost_planned_direction + cost_head_fluctuation + cost_dynamic_objects + cost_update_degree;

	        /*** For visualization **/
            if(if_publish_panel_arrays){
                cost_update_array.data.push_back(cost_update_degree);
                cost_current_velocity_array.data.push_back(cost_current_velocity);
                cost_planned_direction_array.data.push_back(cost_planned_direction);
                cost_head_fluctuation_array.data.push_back(cost_head_fluctuation);
                cost_dynamic_objects_array.data.push_back(cost_dynamic_objects);
                cost_total_array.data.push_back(cost_total_temp);
            }
            /** End of visualization **/

            if(cost_total_temp < min_head_plan_cost)
            {
                min_head_plan_cost = cost_total_temp;
                yaw_to_send = head_yaw_plan_temp;
            }
        }
        double head_planning_update_time = ros::Time::now().toSec() - head_planning_start_time;
        ROS_INFO_THROTTLE(3.0, "head planning algorithm time is: %lf", head_planning_update_time);

        /*** For visualization **/
        if(if_publish_panel_arrays){
            cost_head_update_degree_pub.publish(cost_update_array);
            cost_head_velocity_pub.publish(cost_current_velocity_array);
            cost_head_direction_pub.publish(cost_planned_direction_array);
            cost_head_objects_pub.publish(cost_dynamic_objects_array);
            cost_head_fluctuation_pub.publish(cost_head_fluctuation_array);
            cost_head_final_pub.publish(cost_total_array);
        }

        /** End of visualization **/
        sendMotorCommands(yaw_to_send); //send to motor

        last_head_yaw_plan = yaw_to_send;
    }
}

void setPointSend(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Vector3d a){

    static Eigen::Vector3f p_store_for_em;
    
    static bool init_flag = true;
    if(init_flag){
        p_store_for_em << p0(0), p0(1), p0(2);
        init_flag = false;
    }

    static double def_ori = pp.d_ref;
    
    trajectory_msgs::JointTrajectoryPoint pva_setpoint;

    if(safe_trajectory_avaliable){
        pp.d_ref = def_ori;

        float z_p_set, z_v_set, z_a_set;
        if(if_plan_vertical_path){
            z_p_set = p(2);
            z_v_set = v(2);
            z_a_set = a(2);
        }
        else{
            z_p_set = p_goal(2);
            z_v_set = 0.f;
            z_a_set = 0.f;
        } 

        pva_setpoint.positions.push_back(p(0)); //x
        pva_setpoint.positions.push_back(p(1)); //y
        pva_setpoint.positions.push_back(z_p_set); //z
        pva_setpoint.positions.push_back(0);  //yaw

        pva_setpoint.velocities.push_back(v(0));
        pva_setpoint.velocities.push_back(v(1));
        pva_setpoint.velocities.push_back(z_v_set);

        pva_setpoint.accelerations.push_back(a(0));
        pva_setpoint.accelerations.push_back(a(1));
        pva_setpoint.accelerations.push_back(z_a_set);

        p_store_for_em << p0(0), p0(1), p0(2);
    }else{  //emergency stop
        pp.d_ref = 0.5;
        pva_setpoint.positions.push_back(p_store_for_em(0)); //x
        pva_setpoint.positions.push_back(p_store_for_em(1)); //y
        pva_setpoint.positions.push_back(p_store_for_em(2)); //z
        pva_setpoint.positions.push_back(0);  //yaw

        pva_setpoint.velocities.push_back(0);
        pva_setpoint.velocities.push_back(0);
        pva_setpoint.velocities.push_back(0);

        pva_setpoint.accelerations.push_back(0);
        pva_setpoint.accelerations.push_back(0);
        pva_setpoint.accelerations.push_back(0);

    }

    pva_pub.publish(pva_setpoint);
}

void positionCallback(const geometry_msgs::PoseStamped& msg)
{
    if(!state_locked)
    {
        state_updating = true;

        /** Change from ENU to NWU, NEEDS CAREFUL CHECKING!!!!, chg**/
        p0(0) = msg.pose.position.y;
        p0(1) = -msg.pose.position.x;
        p0(2) = msg.pose.position.z;

        quad.x() = msg.pose.orientation.x;
        quad.y() = msg.pose.orientation.y;   
        quad.z() = msg.pose.orientation.z;
        quad.w() = msg.pose.orientation.w;

        Eigen::Quaternionf axis;
        axis.w() = cos(-PI_2/2.0);
        axis.x() = 0.0;
        axis.y() = 0.0;
        axis.z() = sin(-PI_2/2.0);
        quad = quad * axis;

        // Queue for synchronization
        pose_timestamp_queue.push(msg.header.stamp.toSec());
        pos_queue.push(p0);
        att_queue.push(quad);

        if(pose_timestamp_queue.size()>200){
            pose_timestamp_queue.pop();
            pos_queue.pop();
            att_queue.pop();
        }

        /// Update yaw0 here, should be among [-PI, PI] 
        yaw0 = atan2(2*(quad.w()*quad.z()+quad.x()*quad.y()), 1-2*(quad.z()*quad.z()+quad.y()*quad.y()));   /// TODO: CHECK if this is right!!!

        state_updating = false;
    }

    if(!offboard_ready){
        init_p = p0;  //record a base point
        p_goal(0) = p_goal_raw(0) + init_p(0);
        p_goal(1) = p_goal_raw(1) + init_p(1);
        p_goal(2) = init_p(2);
        ROS_INFO_THROTTLE(1.0, "Waiting for offboard mode. p_goal = %f, %f, %f", p_goal(0), p_goal(1), p_goal(2));
    }else{
        ROS_INFO_THROTTLE(1.0, "IN offboard mode. pgoal = %f, %f, %f", p_goal(0), p_goal(1), p_goal(2));
    }

    ROS_INFO_THROTTLE(1.0, "Current pose. p0 = %f, %f, %f", p0(0), p0(1), p0(2));
}

void velocityCallback(const geometry_msgs::TwistStamped& msg)
{
    if(!state_locked)
    {
        state_updating = true;

        /** Change from ENU to NWU**/
        v0(0) = msg.twist.linear.y;
        v0(1) = -msg.twist.linear.x;
        v0(2) = msg.twist.linear.z;
        yaw0_rate = msg.twist.angular.z;

    	if(fabs(v0(0)) < 0.05) v0(0) = 0.0;  //add a dead zone for v0 used in motion primatives
    	if(fabs(v0(1)) < 0.05) v0(1) = 0.0;  //add a dead zone for v0 used in motion primatives
    	if(fabs(v0(2)) < 0.05) v0(2) = 0.0;  //add a dead zone for v0 used in motion primatives

    	/** Calculate virtual accelerates from velocity. Original accelerates given by px4 is too noisy **/
        static bool init_v_flag = true;
    	static double last_time, last_vx, last_vy, last_vz;

    	if(init_v_flag){
    		init_v_flag = false;
    	}
    	else{
    		double delt_t = ros::Time::now().toSec() - last_time;
    		a0(0) = (v0(0) - last_vx) / delt_t;
    		a0(1) = (v0(1) - last_vy) / delt_t;
    		a0(2) = (v0(2) - last_vz) / delt_t;

    		if(fabs(a0(0)) < 0.1) a0(0) = 0.0;  //dead zone for acc x
     		if(fabs(a0(1)) < 0.1) a0(1) = 0.0; //dead zone for acc y
    		if(fabs(a0(2)) < 0.1) a0(2) = 0.0; //dead zone for acc z
    	}

        last_time = ros::Time::now().toSec();
    	last_vx = v0(0);
    	last_vy = v0(1);
    	last_vz = v0(2);

        state_updating = false;
    }
}


void simPositionVelocityCallback(const nav_msgs::Odometry &msg)
{
    collected_time = msg.header.stamp.toSec();

    if(!state_locked)
    {
        state_updating = true;

        /** NWU **/
        p0(0) = msg.pose.pose.position.x;
        p0(1) = msg.pose.pose.position.y;
        p0(2) = msg.pose.pose.position.z;

        quad.x() = msg.pose.pose.orientation.x;
        quad.y() = msg.pose.pose.orientation.y;
        quad.z() = msg.pose.pose.orientation.z;
        quad.w() = msg.pose.pose.orientation.w;

        pose_timestamp_queue.push(msg.header.stamp.toSec());
        pos_queue.push(p0);
        att_queue.push(quad);

        static bool init_p_flag = true;
        if(init_p_flag){
            init_p = p0;
            init_p_flag = false;
        }
        yaw0 = atan2(2*(quad.w()*quad.z()+quad.x()*quad.y()), 1-2*(quad.z()*quad.z()+quad.y()*quad.y()));

        /*** velocity ***/
        /** NWU**/
        v0(0) = msg.twist.twist.linear.x;
        v0(1) = msg.twist.twist.linear.y;
        v0(2) = msg.twist.twist.linear.z;
        yaw0_rate = msg.twist.twist.angular.z;

        if(fabs(v0(0)) < 0.05) v0(0) = 0.0;  //add a dead zone for v0 used in motion primatives
        if(fabs(v0(1)) < 0.05) v0(1) = 0.0;  //add a dead zone for v0 used in motion primatives
        if(fabs(v0(2)) < 0.05) v0(2) = 0.0;  //add a dead zone for v0 used in motion primatives

        /** Calculate virtual accelerates from velocity. Original accelerates given by px4 is too noisy **/
        static bool init_v_flag = true;
        static double last_time, last_vx, last_vy, last_vz;
        
        if(init_v_flag){
            init_v_flag = false;
        }
        else{
            double delt_t = ros::Time::now().toSec() - last_time;
            a0(0) = (v0(0) - last_vx) / delt_t;
            a0(1) = (v0(1) - last_vy) / delt_t;
            a0(2) = (v0(2) - last_vz) / delt_t;

            if(fabs(a0(0)) < 0.2 || !isfinite(a0(0))) a0(0) = 0.0;  //dead zone for acc x
            if(fabs(a0(1)) < 0.2 || !isfinite(a0(1))) a0(1) = 0.0; //dead zone for acc y
            if(fabs(a0(2)) < 0.2 || !isfinite(a0(2))) a0(2) = 0.0; //dead zone for acc z
        }

        last_time = ros::Time::now().toSec();
        last_vx = v0(0);
        last_vy = v0(1);
        last_vz = v0(2);

        state_updating = false;
    }
}


void uavModeCallback(const mavros_msgs::State &msg)
{
    if(msg.mode=="OFFBOARD") offboard_ready=true;
    else offboard_ready=false;
}

void sendMotorCommands(double yaw) // Range[-Pi, Pi], [0, 1]
{
    /** Set a limitation **/
    double delt_yaw = yaw - motor_yaw;
    if(fabs(delt_yaw) > head_max_yaw_delt){
        yaw = motor_yaw + head_max_yaw_delt * delt_yaw / fabs(delt_yaw);
    }

    if(!if_in_simulation){
        static geometry_msgs::Point32 head_cmd;
        head_cmd.x = -yaw + init_head_yaw; 
        head_cmd.y = motor_velocity_set;
        head_cmd_pub.publish(head_cmd);
    }else{
        static std_msgs::Float64 head_cmd;
        head_cmd.data = yaw - init_head_yaw;  
        head_cmd_pub.publish(head_cmd);
    }
    
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_planning_gradient_sample");
    ros::NodeHandle nh;

    setParameters();    /// use the parameters initialized in this function if not excuted by a launch file.
    getParameterList(nh);

    if(if_flyback_enable){
        max_sample_delt_angle = 2.5;  //2.09
    }else{
        max_sample_delt_angle = 1.67;
    }

    trajectory_time_table->csv2pva_table(TABLE_PATH);
    ROS_WARN("trajectory_time_table loaded!");

    // State parameters initiate
    p0 << 0.0, 0.0, 0.0;
    v0 << 0.0, 0.0, 0.0;
    a0 << 0.0, 0.0, 0.0;
    yaw0 = 0.0;

    _direction_update_buffer = Eigen::VectorXf::Zero(LOOKING_PIECES_SIZE);
    heading_resolution =  2.f * PI / (float)LOOKING_PIECES_SIZE;
    mid_seq_num = (int)(LOOKING_PIECES_SIZE / 2); // start from 0, mid is 9 when LOOKING_PIECES_SIZE is 18

    int valid_piece_num = (int)((float)CAMERA_H_FOV / 180.f * PI / heading_resolution);
    if(valid_piece_num % 2 == 0)  valid_piece_num -= 1;
    if(valid_piece_num < 1){
        ROS_ERROR("No enough view field with the current camera set!");
        return 0;
    }
    valid_piece_num_one_side = (valid_piece_num - 1) / 2;

    ROS_INFO("Heading resolution = %f (rad), Fov = %f, valid_piece_num = %d", heading_resolution, CAMERA_H_FOV, valid_piece_num);

    // ringbuffer cloud2
    cloud2_pub = nh.advertise<sensor_msgs::PointCloud2>("/ring_buffer/cloud_ob", 1, true);
    cloud_edf_pub = nh.advertise<sensor_msgs::PointCloud2>("/ring_buffer/edf", 1, true);
    map_center_pub = nh.advertise<geometry_msgs::PointStamped>("/map_center",1,true) ;

    goal_pub = nh.advertise<geometry_msgs::PointStamped>("/goal_position",1,true);
    current_marker_pub = nh.advertise<visualization_msgs::Marker>("/visualization_marker", 1);

    motor_pub = nh.advertise<geometry_msgs::Point32>("/place_velocity_info_corrected", 1, true); 
    pva_pub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("/pva_setpoint", 1, true);

    ros::Subscriber mode_sub, position_isolate_sub, velocity_isolate_sub, motor_sub, cloud_sub, sim_odom_isolate_sub;

    /*** For real world test***/
    if(!if_in_simulation){
        mode_sub = nh.subscribe("/mavros/state", 1, uavModeCallback);
        position_isolate_sub =  nh.subscribe("/mavros/local_position/pose", 1, positionCallback);
        velocity_isolate_sub = nh.subscribe("/mavros/local_position/velocity_local", 1, velocityCallback);

        typedef message_filters::sync_policies::ApproximateTime<control_msgs::JointControllerState,sensor_msgs::PointCloud2> mapSyncPolicy;
        message_filters::Subscriber<control_msgs::JointControllerState> *motor_sub;
        message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
        motor_sub = new message_filters::Subscriber<control_msgs::JointControllerState>(nh, "/place_velocity_info_joint_state", 2);
        cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/d400/depth/color/points", 2);

        message_filters::Synchronizer<mapSyncPolicy>* sync_;
        sync_ = new message_filters::Synchronizer<mapSyncPolicy>(mapSyncPolicy(10), *motor_sub, *cloud_sub);
        sync_->registerCallback(boost::bind(&cloudCallback, _1, _2));

        head_cmd_pub = nh.advertise<geometry_msgs::Point32>("/gimbal_commands", 2, true);
    }else{
         /*** For simulation test***/
        offboard_ready = true;
        sim_odom_isolate_sub = nh.subscribe("/iris/ground_truth/odometry", 1, simPositionVelocityCallback);

        typedef message_filters::sync_policies::ApproximateTime<control_msgs::JointControllerState,sensor_msgs::PointCloud2> mapSyncPolicy;
        message_filters::Subscriber<control_msgs::JointControllerState> *motor_sub;
        message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
        motor_sub = new message_filters::Subscriber<control_msgs::JointControllerState>(nh, "/iris/joint1_position_controller/state", 2);
        cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/iris/vi_sensor/camera_depth/depth/points", 2);

        message_filters::Synchronizer<mapSyncPolicy>* sync_;
        sync_ = new message_filters::Synchronizer<mapSyncPolicy>(mapSyncPolicy(10), *motor_sub, *cloud_sub);
        sync_->registerCallback(boost::bind(&cloudCallback, _1, _2));

        head_cmd_pub = nh.advertise<std_msgs::Float64>("/iris/joint1_position_controller/command", 2, true);
        if(if_sim_lee_position_controller){
            sim_trajectory_pub = nh.advertise<geometry_msgs::Pose>("/trajectory_setpoint", 2, true);
        }
    }

    // timer for publish ringbuffer as pointcloud
    ros::Timer timer1 = nh.createTimer(ros::Duration(0.2), cloudPubCallback); // RATE 5 Hz to publish

    // timer for trajectory generation
    ros::Timer timer2 = nh.createTimer(ros::Duration(SEND_DURATION), trajectoryCallback);

    /***** Publisher for visulization ****/
    cost_head_update_degree_pub = nh.advertise<std_msgs::Float64MultiArray>("/head_cost/cost_head_update",1,true);
    cost_head_velocity_pub = nh.advertise<std_msgs::Float64MultiArray>("/head_cost/cost_head_velocity",1,true);
    cost_head_direction_pub = nh.advertise<std_msgs::Float64MultiArray>("/head_cost/cost_head_direction",1,true);
    cost_head_objects_pub = nh.advertise<std_msgs::Float64MultiArray>("/head_cost/cost_head_objects",1,true);
    cost_head_fluctuation_pub = nh.advertise<std_msgs::Float64MultiArray>("/head_cost/cost_head_fluctuation",1,true);
    cost_head_final_pub = nh.advertise<std_msgs::Float64MultiArray>("/head_cost/cost_head_final",1,true);

    std::cout << "Start mapping!" << std::endl;

    // ros::spin();
    ros::AsyncSpinner spinner(3); // Use 3 threads
    spinner.start();
    ros::waitForShutdown();

    return 0;
}

void getParameterList(ros::NodeHandle nh){
    /*** Read parameter list ***/
    nh.getParam("/local_planning_gradient_sample/goal_position_x", p_goal(0));
    nh.getParam("/local_planning_gradient_sample/goal_position_y", p_goal(1));
    nh.getParam("/local_planning_gradient_sample/goal_position_z", p_goal(2));
    nh.getParam("/local_planning_gradient_sample/MAX_V_XY", MAX_V_XY);
    nh.getParam("/local_planning_gradient_sample/MAX_V_Z_UP", MAX_V_Z_UP);
    nh.getParam("/local_planning_gradient_sample/MAX_V_Z_DOWN", MAX_V_Z_DOWN);
    nh.getParam("/local_planning_gradient_sample/MAX_A", MAX_A);
    nh.getParam("/local_planning_gradient_sample/distance_reference_length", pp.d_ref);
    nh.getParam("/local_planning_gradient_sample/toward_goal_k1_xy", pp.k1_xy);
    nh.getParam("/local_planning_gradient_sample/toward_goal_k1_z", pp.k1_z);
    nh.getParam("/local_planning_gradient_sample/less_variation_k2_xy", pp.k2_xy);
    nh.getParam("/local_planning_gradient_sample/less_variation_k2_z", pp.k2_z);
    nh.getParam("/local_planning_gradient_sample/observation_direction_k3_xy", pp.k3_xy);
    nh.getParam("/local_planning_gradient_sample/observation_direction_k3_z", pp.k3_z);

    nh.getParam("/local_planning_gradient_sample/v_max_at_goal_ori", pp.v_max_at_goal_ori);
    nh.getParam("/local_planning_gradient_sample/collision_threshold_static", pp.collision_threshold_static);
    nh.getParam("/local_planning_gradient_sample/k_current_v", hp.k_current_v);
    nh.getParam("/local_planning_gradient_sample/k_planned_direction", hp.k_planned_direction);
    nh.getParam("/local_planning_gradient_sample/k_v_fluctuation", hp.k_v_fluctuation);
    nh.getParam("/local_planning_gradient_sample/k_common_update", hp.k_common_update);
    nh.getParam("/local_planning_gradient_sample/if_publish_panel_arrays", if_publish_panel_arrays);
    nh.getParam("/local_planning_gradient_sample/CAMERA_H_FOV", CAMERA_H_FOV);

    nh.getParam("/local_planning_gradient_sample/head_max_yaw_delt", head_max_yaw_delt);
    nh.getParam("/local_planning_gradient_sample/motor_velocity_set", motor_velocity_set);
    nh.getParam("/local_planning_gradient_sample/if_plan_vertical_path", if_plan_vertical_path);
    nh.getParam("/local_planning_gradient_sample/if_in_simulation", if_in_simulation);
    nh.getParam("/local_planning_gradient_sample/if_flyback_enable", if_flyback_enable);
    // nh.getParam("/local_planning_gradient_sample/HEIGHT_LIMIT", HEIGHT_LIMIT);
    // nh.getParam("/local_planning_gradient_sample/XY_LIMIT", XY_LIMIT);
    nh.getParam("/local_planning_gradient_sample/PLAN_INIT_STATE_CHANGE_THRESHOLD", PLAN_INIT_STATE_CHANGE_THRESHOLD);
    nh.getParam("/local_planning_gradient_sample/TABLE_PATH", TABLE_PATH);

    p_goal_raw = p_goal;
    if(if_in_simulation)  ROS_WARN("In simulation mode");

    ROS_INFO("Parameters list reading finished! Goal position is: (%f, %f, %f), MAX Vel is %f", p_goal(0), p_goal(1), p_goal(2), MAX_V_XY);
}

void setParameters(){
    p_goal(0) = 0.0;
    p_goal(1)=15.0;
    p_goal(2)=1.0;
    p_goal_raw = p_goal;

    MAX_V_XY = 3.0;
    MAX_V_Z_UP = 1.0;
    MAX_V_Z_DOWN = 1.0;
    MAX_A = 4.0;
    pp.d_ref = 2.0;
    pp.k1_xy = 4.0;
    pp.k1_z = 4.0;
    pp.k2_xy = 2.0;
    pp.k2_z = 2.0;
    pp.k3_xy = 0.5;
    pp.k3_z = 0.5;
    pp.v_max_at_goal_ori = 1.5;
    pp.collision_threshold_static = 0.6;
    hp.k_current_v = 0.7;
    hp.k_planned_direction = 0.2;
    hp.k_v_fluctuation = 0.4;
    hp.k_common_update = 0.1;
    if_publish_panel_arrays = true;
    CAMERA_H_FOV = 62;

    head_max_yaw_delt = 0.5;
    motor_velocity_set = 50;
    if_plan_vertical_path = true;
    if_in_simulation = false;
    if_flyback_enable = false;
    HEIGHT_LIMIT = 1.8;
    XY_LIMIT = 15.0;
    PLAN_INIT_STATE_CHANGE_THRESHOLD = 0.15;

    TABLE_PATH = "/home/topup/chg_ws/src/tables/p3-84_v3_a4_res0-1.csv";
}


// Created by clarence on 19-8-29.
//

