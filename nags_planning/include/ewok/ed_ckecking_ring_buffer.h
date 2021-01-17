/**
* This file is modified from Ewok.
*
* Ewok is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Ewok is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with Ewok. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef EWOK_RING_BUFFER_INCLUDE_EWOK_ED_NOR_RING_BUFFER_H_
#define EWOK_RING_BUFFER_INCLUDE_EWOK_ED_NOR_RING_BUFFER_H_

#include <ewok/raycast_ring_buffer.h>

#include <deque>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <iostream>

namespace ewok
{
template <int _POW, typename _Datatype = int16_t, typename _Scalar = float, typename _Flag = uint8_t>
class EuclideanDistanceCheckingRingBuffer
{
  public:
    static const int _N = (1 << _POW);  // 2 to the power of POW

    // Other definitions
    typedef Eigen::Matrix<_Scalar, 4, 1> Vector4;
    typedef Eigen::Matrix<_Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<int, 3, 1> Vector3i;
    typedef std::vector<Vector4, Eigen::aligned_allocator<Vector4>> PointCloud;

    typedef std::shared_ptr<EuclideanDistanceCheckingRingBuffer<_POW, _Datatype, _Scalar, _Flag>> Ptr;

    EuclideanDistanceCheckingRingBuffer(const _Scalar &resolution, const _Scalar &truncation_distance)
      : resolution_(resolution)
      , truncation_distance_(truncation_distance)
      , occupancy_buffer_(resolution)
      , occupancy_buffer_dynamic_(resolution)
      , tmp_buffer1_(resolution)
      , tmp_buffer2_(resolution)
      , distance_buffer_(resolution, truncation_distance)
    {
        distance_buffer_.setEmptyElement(std::numeric_limits<_Scalar>::max());
    }

    inline void getIdx(const Vector3 &point, Vector3i &idx) const { distance_buffer_.getIdx(point, idx); }

    inline void getPoint(const Vector3i &idx, Vector3 &point) const { distance_buffer_.getPoint(idx, point); }

    inline Vector3i getVolumeCenter() { return distance_buffer_.getVolumeCenter(); }

    void updateDistance() { compute_edt3d(); }

    void updateDistanceDynamic(const PointCloud &cloud, const Vector3 &origin) { compute_edt3d_dynamic(cloud, origin); } // CHG

    void insertPointCloud(const PointCloud &cloud, const Vector3 &origin)
    {
        occupancy_buffer_.insertPointCloud(cloud, origin);
    }

    void removePointCloud(const PointCloud &cloud, const Vector3 &origin)  // CHG
    {
        occupancy_buffer_.removePointCloud(cloud, origin);
    }

    void insertPointCloudDynamic(const PointCloud &cloud, const Vector3 &origin)  // CHG
    {
        occupancy_buffer_dynamic_.insertPointCloudDynamic(cloud, origin);
    }

    void removePointCloudDynamic(const PointCloud &cloud, const Vector3 &origin)  // CHG
    {
        occupancy_buffer_dynamic_.removePointCloudDynamic(cloud, origin);
    }


    // Add offset
    virtual void setOffset(const Vector3i &off)
    {
        occupancy_buffer_.setOffset(off);
        occupancy_buffer_dynamic_.setOffset(off);
        distance_buffer_.setOffset(off);
    }

    // Add  moveVolume
    virtual void moveVolume(const Vector3i &direction)
    {
        occupancy_buffer_.moveVolume(direction);
        occupancy_buffer_dynamic_.moveVolume(direction);
        distance_buffer_.moveVolume(direction);
    }

    // get ringbuffer as pointcloud
    void getBufferAsCloud(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center)
    {
        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        distance_buffer_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only occupied voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        Vector3 p;
                        getPoint(coord, p);
                        pcl::PointXYZ pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);
                       
                        cloud.points.push_back(pclp);
                    }
                }
            }
        }
    }

    // get occupied and free voxels in ringbuffer as pointcloud
    void getBufferAllAsCloud(pcl::PointCloud<pcl::PointXYZI> &cloud, Eigen::Vector3d &center)
    {
        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        distance_buffer_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only occupied voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        Vector3 p;
                        getPoint(coord, p);
                        pcl::PointXYZI pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);
                        pclp.intensity = 1.0;

                        cloud.points.push_back(pclp);
                    }else if(occupancy_buffer_.isFree(coord)){
                        Vector3 p;
                        getPoint(coord, p);
                        pcl::PointXYZI pclp;
                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);
                        pclp.intensity = 0.5;

                        cloud.points.push_back(pclp);
                    }
                }
            }
        }
    }


    // get ringbuffer free space pointcloud
    void getBufferFSCloud(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center)
    {
        // get center of ring buffer
        Vector3i c_idx = getVolumeCenter();
        Vector3 ct;
        getPoint(c_idx, ct);
        center(0) = ct(0);
        center(1) = ct(1);
        center(2) = ct(2);

        // convert ring buffer to point cloud
        Vector3i off;
        distance_buffer_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only free voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                                       
                    if(occupancy_buffer_.isFree(coord))
                    {

                        Vector3 p;
                        getPoint(coord, p); 
                        pcl::PointXYZ pclp;

                        pclp.x = p(0);
                        pclp.y = p(1);
                        pclp.z = p(2);
                        cloud.points.push_back(pclp);
                    }
                   
                }
            }
        }
    }

    bool collision_checking_no_fence(Eigen::Vector3f *traj_points, int Num, _Scalar threshold) {
        bool all_safe = true;

        //Check the last point in the path first
        Vector3i traj_point_idx_the_last_point;
        Vector3 traj_point_the_last_point = traj_points[Num-1].template cast<_Scalar>();
        distance_buffer_.getIdx(traj_point_the_last_point, traj_point_idx_the_last_point);
        if (distance_buffer_.insideVolume(traj_point_idx_the_last_point)){
            if(distance_buffer_.at(traj_point_idx_the_last_point) < threshold ){
                return false;
            }
        }else{
            return false;
        }

        // Set first late checked point
        Vector3i traj_point_idx_last;
        Vector3 traj_point_last = traj_points[2].template cast<_Scalar>();
        distance_buffer_.getIdx(traj_point_last, traj_point_idx_last);
        float distance_last_point;
        if (distance_buffer_.insideVolume(traj_point_idx_last)){
            distance_last_point = distance_buffer_.at(traj_point_idx_last);
        }

        for (int i = 3; i < Num; i+=2) {  /// Start from the third point, check every two point
            Vector3 traj_point = traj_points[i].template cast<_Scalar>();
            Vector3i traj_point_idx;

            distance_buffer_.getIdx(traj_point, traj_point_idx);

            if (distance_buffer_.insideVolume(traj_point_idx)) {  //if inside
                float distance_this = distance_buffer_.at(traj_point_idx);
//                std::cout << distance_this <<", ";
                if (distance_this > threshold || distance_this >= distance_last_point) {
                    distance_last_point = distance_this;
                    traj_point_last = traj_point;
                    continue;
                }else{
                    all_safe = false;
                    break;
                }

            } else {
                all_safe = false;
//                std::cout << "not safe because out of volume" << std::endl;
                break;
            }
        }
//        std::cout << "Num=" << Num << std::endl;
        return all_safe;
    }


    bool collision_checking_no_fence_min_dist(Eigen::Vector3f *traj_points, int Num, float &min_dist) {
        min_dist = 1000.f;
        bool far_from_obstacles = true;

        // Set first late checked point
        Vector3i traj_point_idx_last;
        Vector3 traj_point_last = traj_points[1].template cast<_Scalar>();
        distance_buffer_.getIdx(traj_point_last, traj_point_idx_last);
        float distance_last_point;
        if (distance_buffer_.insideVolume(traj_point_idx_last)){
            distance_last_point = distance_buffer_.at(traj_point_idx_last);
            if(distance_last_point < min_dist){
                min_dist = distance_last_point;
            }
        }
        for (int i = 2; i < Num; i+=1) {
            Vector3 traj_point = traj_points[i].template cast<_Scalar>();
            Vector3i traj_point_idx;

            distance_buffer_.getIdx(traj_point, traj_point_idx);

            if (distance_buffer_.insideVolume(traj_point_idx)) {  //if inside
                float distance_this = distance_buffer_.at(traj_point_idx);
                if(distance_this < min_dist){
                    min_dist = distance_this;
                }
                if (distance_this < distance_last_point) {
                    far_from_obstacles = false;
                }
                distance_last_point = distance_this;
            }
        }
        return far_from_obstacles;
    }

    int collision_checking_with_fence(Eigen::Vector3f *traj_points, int Num, _Scalar threshold, Eigen::Vector3d init_position, float max_height=120.f, float xy_max=10000.f) {
        int all_safe = 1;

        //Check the last point in the path first
        Vector3i traj_point_idx_the_last_point;
        Vector3 traj_point_the_last_point = traj_points[Num-1].template cast<_Scalar>();

        distance_buffer_.getIdx(traj_point_the_last_point, traj_point_idx_the_last_point);
        if (occupancy_buffer_.isFree(traj_point_idx_the_last_point)){ //(distance_buffer_.insideVolume(traj_point_idx_the_last_point)){
            if(distance_buffer_.at(traj_point_idx_the_last_point) < threshold ){
                return -1;
            }
        }else{
            return -2;
        }

        // Set first late checked point
        Vector3i traj_point_idx_last;
        Vector3 traj_point_last = traj_points[2].template cast<_Scalar>();
        distance_buffer_.getIdx(traj_point_last, traj_point_idx_last);
        float distance_last_point;
        if (distance_buffer_.insideVolume(traj_point_idx_last)){
            distance_last_point = distance_buffer_.at(traj_point_idx_last);
        }

        for (int i = 3; i < Num; i+=2) {  /// Start from the third point, check every two point
            Vector3 traj_point = traj_points[i].template cast<_Scalar>();
            Vector3i traj_point_idx;

            if(fabs(traj_point(0)-init_position(0)) > xy_max && fabs(traj_point(0)-init_position(0)) > fabs(traj_point_last[0]-init_position(0))){
                all_safe = -3;
                break;
            }else if(fabs(traj_point(1)-init_position(1)) > xy_max && fabs(traj_point(1)-init_position(1)) > fabs(traj_point_last[1]-init_position(1))){
                all_safe = -4;
                break;
            }else if(traj_point(2)-init_position(2) > max_height && traj_point[2] > traj_point_last[2]){  /// Height limitation
                all_safe = -5;
                break;
            }else if(traj_point(2) - init_position(2) < -0.6f){
                all_safe = -6;
                break;
            }

            distance_buffer_.getIdx(traj_point, traj_point_idx);

            if(occupancy_buffer_.isFree(traj_point_idx)){ //(distance_buffer_.insideVolume(traj_point_idx)) {  //if inside
                float distance_this = distance_buffer_.at(traj_point_idx);
                if (distance_this > threshold || distance_this >= distance_last_point) {
                    distance_last_point = distance_this;
                    traj_point_last = traj_point;
                    continue;
                }else{
                    all_safe = -7;
                    break;
                }
                
            } else {
                all_safe = -8;
//                std::cout << "not safe because out of volume" << std::endl;
                break;
            }
        }
//        std::cout << "Num=" << Num << std::endl;
        return all_safe;
    }


    bool collision_checking_with_fence_min_dist(Eigen::Vector3f *traj_points, int Num, float &min_dist, Eigen::Vector3d init_position, float max_height=120.f, float xy_max=10000.f) {
        min_dist = 1000.f;
        bool far_from_obstacles = true;

        // Set first late checked point
        Vector3i traj_point_idx_last;
        Vector3 traj_point_last = traj_points[1].template cast<_Scalar>();
        distance_buffer_.getIdx(traj_point_last, traj_point_idx_last);
        float distance_last_point;
        if (distance_buffer_.insideVolume(traj_point_idx_last)){
            distance_last_point = distance_buffer_.at(traj_point_idx_last);
            if(distance_last_point < min_dist){
                min_dist = distance_last_point;
            }
        }
        for (int i = 2; i < Num; i+=1) {
            Vector3 traj_point = traj_points[i].template cast<_Scalar>();
            Vector3i traj_point_idx;

            if(fabs(traj_point(0)-init_position(0)) > xy_max && fabs(traj_point(0)-init_position(0)) > fabs(traj_point_last[0]-init_position(0))){
                min_dist = 0;
                far_from_obstacles = false;
                break;
            }else if(fabs(traj_point(1)-init_position(1)) > xy_max && fabs(traj_point(1)-init_position(1)) > fabs(traj_point_last[1]-init_position(1))){
                min_dist = 0;
                far_from_obstacles = false;
                break;
            }else if(traj_point(2)-init_position(2) > max_height && traj_point[2] > traj_point_last[2]){  /// Height limitation
                min_dist = 0;
                far_from_obstacles = false;
                break;
            }else if(traj_point(2) - init_position(2) < -0.6f){
                min_dist = 0;
                far_from_obstacles = false;
                break;
            }

            distance_buffer_.getIdx(traj_point, traj_point_idx);
            if (distance_buffer_.insideVolume(traj_point_idx)) {  //if inside
                float distance_this = distance_buffer_.at(traj_point_idx);

                if(distance_this < min_dist){
                    min_dist = distance_this;
                }
                if (distance_this < distance_last_point) {
                    far_from_obstacles = false;
                }
                distance_last_point = distance_this;
            }
        }

        return far_from_obstacles;
    }

    Vector3i get_rgb_edf(float x, float y, float z)
    {
        Vector3 cloud_point;
        Vector3i cloud_point_idx;
        cloud_point(0) = (_Scalar)x;
        cloud_point(1) = (_Scalar)y;
        cloud_point(2) = (_Scalar)z;

        distance_buffer_.getIdx(cloud_point, cloud_point_idx);
        float distance = (float)distance_buffer_.at(cloud_point_idx);

        int value = floor(distance * 240); // Mapping 0~1.0 to 0~240
        value = value > 240 ? 240 : value;

        // 240 degrees are divided into 4 sections, 0 is Red-dominant, 1 and 2 are Green-dominant,
        // 3 is Blue-dominant. The dorminant color is 255 and another color is always 0 while the
        // remaining color increases or decreases between 0~255
        int section = value / 60;
        float float_key = (value % 60) / (float)60 * 255;
        int key = floor(float_key);
        int nkey = 255 - key;

        Vector3i point_RGB;

        switch(section) {
            case 0: // G increase
                point_RGB(0) = 255;
                point_RGB(1) = key;
                point_RGB(2) = 0;
                break;
            case 1: // R decrease
                point_RGB(0) = nkey;
                point_RGB(1) = 255;
                point_RGB(2) = 0;
                break;
            case 2: // B increase
                point_RGB(0) = 0;
                point_RGB(1) = 255;
                point_RGB(2) = key;
                break;
            case 3: // G decrease
                point_RGB(0) = 0;
                point_RGB(1) = nkey;
                point_RGB(2) = 255;
                break;
            case 4:
                point_RGB(0) = 0;
                point_RGB(1) = 0;
                point_RGB(2) = 255;
                break;
            default: // White
                point_RGB(0) = 255;
                point_RGB(1) = 255;
                point_RGB(2) = 255;
        }

        return point_RGB;
    }



    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
    void compute_edt3d()
    {
        Vector3i offset;
        distance_buffer_.getOffset(offset); //off_set in distance buffer, integer, sequence of distance ring buffer

        Vector3i min_vec, max_vec; // To store the max and min index of the points whose distance value is required
        occupancy_buffer_.getUpdatedMinMax(min_vec, max_vec); // min_vec = off_set in occupancy buffer + (N-1), max_vec = off_set in occupancy buffer

        min_vec -= offset;  // min_vec = off_set in occupancy buffer - off_set in distance buffer + (N-1)
        max_vec -= offset;  // max_vec = off_set in occupancy buffer - off_set in distance buffer

        min_vec.array() -= truncation_distance_ / resolution_; // 1.0 / 0.2 = 5, truncation_distance_ is set by user, 1.0 in local_planning.cpp
        max_vec.array() += truncation_distance_ / resolution_; // Shrink the computing area

        min_vec.array() = min_vec.array().max(Vector3i(0, 0, 0).array());  // Set a limit, in case the calculated min_vec is too large
        max_vec.array() = max_vec.array().min(Vector3i(_N - 1, _N - 1, _N - 1).array());

        // ROS_INFO_STREAM("min_vec: " << min_vec.transpose() << " max_vec: " << max_vec.transpose());

        for(int x = min_vec[0]; x <= max_vec[0]; x++)  //x, y plane search
        {
            for(int y = min_vec[1]; y <= max_vec[1]; y++)
            {
                fill_edt(
                    [&](int z) {
                        return occupancy_buffer_.isOccupied(offset + Vector3i(x, y, z)) ?
                                   0 :
                                   std::numeric_limits<_Scalar>::max(); // _Scalar: float
                    },
                    [&](int z, _Scalar val) { tmp_buffer1_.at(Vector3i(x, y, z)) = val; }, min_vec[2], max_vec[2]);
            }
        }

        for(int x = min_vec[0]; x <= max_vec[0]; x++)  // x, z plane search
        {
            for(int z = min_vec[2]; z <= max_vec[2]; z++)
            {
                fill_edt([&](int y) { return tmp_buffer1_.at(Vector3i(x, y, z)); },
                         [&](int y, _Scalar val) { tmp_buffer2_.at(Vector3i(x, y, z)) = val; }, min_vec[1], max_vec[1]);
            }
        }

        for(int y = min_vec[1]; y <= max_vec[1]; y++) // y, z plane search
        {
            for(int z = min_vec[2]; z <= max_vec[2]; z++)
            {
                fill_edt([&](int x) { return tmp_buffer2_.at(Vector3i(x, y, z)); },
                         [&](int x, _Scalar val) {
                             distance_buffer_.at(offset + Vector3i(x, y, z)) =
                                 std::min(resolution_ * std::sqrt(val), truncation_distance_);  // Set final offset
                         },
                         min_vec[0], max_vec[0]);
            }
        }

        occupancy_buffer_.clearUpdatedMinMax();
    }

    void compute_edt3d_dynamic(const PointCloud &cloud, const Vector3 &origin) // CHG, distance field calculation
    {
        // Copy occupancy_buffer_ to occupancy_buffer_dynamic_ first


        // convert ring buffer to point cloud to copy by inserting, which is much faster than using "=" directly on class member
        PointCloud static_cloud;
        Vector3i off;
        distance_buffer_.getOffset(off);
        for(int x = 0; x < _N; x++)
        {
            for(int y = 0; y < _N; y++)
            {
                for(int z = 0; z < _N; z++)
                {
                    // only occupied voxel is return
                    Vector3i coord(x, y, z);
                    coord += off;
                    if(occupancy_buffer_.isOccupied(coord))
                    {
                        Vector3 p;
                        getPoint(coord, p);
                        static_cloud.push_back(Eigen::Vector4f(p(0), p(1), p(2), 0));
                    }
                }
            }
        }

        insertPointCloudDynamic(static_cloud, origin); //CHG
        insertPointCloudDynamic(cloud, origin); //CHG

        Vector3i offset;
        distance_buffer_.getOffset(offset); //off_set in distance buffer, integer, sequence of distance ring buffer

        Vector3i min_vec, max_vec; // To store the max and min index of the points whose distance value is required
        occupancy_buffer_dynamic_.getUpdatedMinMax(min_vec, max_vec); // min_vec = off_set in occupancy buffer + (N-1), max_vec = off_set in occupancy buffer

        min_vec -= offset;  // min_vec = off_set in occupancy buffer - off_set in distance buffer + (N-1)
        max_vec -= offset;  // max_vec = off_set in occupancy buffer - off_set in distance buffer

        min_vec.array() -= truncation_distance_ / resolution_; // 1.0 / 0.2 = 5, truncation_distance_ is set by user, 1.0 in local_planning.cpp
        max_vec.array() += truncation_distance_ / resolution_; // Shrink the computing area

        min_vec.array() = min_vec.array().max(Vector3i(0, 0, 0).array());  // Set a limit, in case the calculated min_vec is too large
        max_vec.array() = max_vec.array().min(Vector3i(_N - 1, _N - 1, _N - 1).array());

        // ROS_INFO_STREAM("min_vec: " << min_vec.transpose() << " max_vec: " << max_vec.transpose());

        for(int x = min_vec[0]; x <= max_vec[0]; x++)  //x, y plane search
        {
            for(int y = min_vec[1]; y <= max_vec[1]; y++)
            {
                fill_edt(
                        [&](int z) {
                            return occupancy_buffer_dynamic_.isOccupied(offset + Vector3i(x, y, z)) ?
                                   0 :
                                   std::numeric_limits<_Scalar>::max(); // _Scalar: float
                        },
                        [&](int z, _Scalar val) { tmp_buffer1_.at(Vector3i(x, y, z)) = val; }, min_vec[2], max_vec[2]);
            }
        }

        for(int x = min_vec[0]; x <= max_vec[0]; x++)  // x, z plane search
        {
            for(int z = min_vec[2]; z <= max_vec[2]; z++)
            {
                fill_edt([&](int y) { return tmp_buffer1_.at(Vector3i(x, y, z)); },
                         [&](int y, _Scalar val) { tmp_buffer2_.at(Vector3i(x, y, z)) = val; }, min_vec[1], max_vec[1]);
            }
        }

        for(int y = min_vec[1]; y <= max_vec[1]; y++) // y, z plane search
        {
            for(int z = min_vec[2]; z <= max_vec[2]; z++)
            {
                fill_edt([&](int x) { return tmp_buffer2_.at(Vector3i(x, y, z)); },
                         [&](int x, _Scalar val) {
                             distance_buffer_.at(offset + Vector3i(x, y, z)) =
                                     std::min(resolution_ * std::sqrt(val), truncation_distance_);  // Set final offset
                         },
                         min_vec[0], max_vec[0]);
            }
        }
        removePointCloudDynamic(static_cloud, origin); //CHG
        removePointCloudDynamic(cloud, origin); //CHG
        occupancy_buffer_dynamic_.clearUpdatedMinMax();
    }


    template <typename F_get_val, typename F_set_val>
    void fill_edt(F_get_val f_get_val, F_set_val f_set_val, int start = 0, int end = _N - 1) 
    {
        int v[_N];
        _Scalar z[_N + 1];

        int k = start;
        v[start] = start;
        z[start] = -std::numeric_limits<_Scalar>::max();
        z[start + 1] = std::numeric_limits<_Scalar>::max();

        for(int q = start + 1; q <= end; q++)
        {
            k++;
            _Scalar s;

            do
            {
                k--;
                s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
                // ROS_INFO_STREAM("k: " << k << " s: " <<  s  << " z[k] " << z[k] << " v[k] " << v[k]);

            } while(s <= z[k]);

            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = std::numeric_limits<_Scalar>::max();
        }

        k = start;

        for(int q = start; q <= end; q++)
        {
            while(z[k + 1] < q) k++;
            _Scalar val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
            //      if(val < std::numeric_limits<_Scalar>::max())
            //  ROS_INFO_STREAM("val: " << val << " q: " << q << " v[k] " << v[k]);
            // if(val > truncation_distance_*truncation_distance_) val = std::numeric_limits<_Scalar>::max();
            f_set_val(q, val);
        }
    }

    _Scalar resolution_;
    _Scalar truncation_distance_;

    RaycastRingBuffer<_POW, _Datatype, _Scalar, _Flag> occupancy_buffer_;

    RaycastRingBuffer<_POW, _Datatype, _Scalar, _Flag> occupancy_buffer_dynamic_;

    RingBufferBase<_POW, _Scalar, _Scalar> distance_buffer_; 

    RingBufferBase<_POW, _Scalar, _Scalar> tmp_buffer1_, tmp_buffer2_;

};
}

#endif  // EWOK_RING_BUFFER_INCLUDE_EWOK_ED_RING_BUFFER_H_
