//
// Created by cc on 2019/12/3.
//
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/Float64MultiArray.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>

std::map<std::string, std::vector<double>> painting_data_map;

void rotateVector(cv::Point &center, cv::Point &start_point, float angle, cv::Point &end_point)
{
    cv::Point start_vector = start_point - center;
    cv::Point new_point_vector;
    new_point_vector.x = cos(angle)*start_vector.x + sin(angle)*start_vector.y;
    new_point_vector.y = -sin(angle)*start_vector.x + cos(angle)*start_vector.y;
    end_point = new_point_vector + center;
}

void costHeadUpdateCallback(const std_msgs::Float64MultiArray &msg)
{
    painting_data_map["costHeadUpdate"]=msg.data;
}

void costHeadObjectsCallback(const std_msgs::Float64MultiArray &msg)
{
    painting_data_map["costHeadObjects"]=msg.data;
}

void costHeadVelocityCallback(const std_msgs::Float64MultiArray &msg)
{
    painting_data_map["costHeadVelocity"]=msg.data;
}

void costHeadDirectionCallback(const std_msgs::Float64MultiArray &msg)
{
    painting_data_map["costHeadDirection"]=msg.data;
}

void costHeadFluctuationCallback(const std_msgs::Float64MultiArray &msg)
{
    painting_data_map["costHeadFluctuation"]=msg.data;
}

void costHeadFinalCallback(const std_msgs::Float64MultiArray &msg)
{
    painting_data_map["costHeadFinal"]=msg.data;
}

void displayTimer(const ros::TimerEvent& e)
{
    int rows = 480;  // y
    int cols = 700;  // x
    int center_step = 200;
    int size_one_pannel = 60;
    cv::Mat background = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::Point center = cv::Point(center_step/2, center_step/2);

    for(auto & vector_i : painting_data_map){
        int num = vector_i.second.size();
        float angle_one_piece = 2*M_PI/num;

        /** Map color to 0, 255 **/
        double min_value = 1000000.f;
        double max_value = -1000000.f;
        for(auto & value_i : vector_i.second){

            if(value_i < min_value){
                min_value = value_i;
            }
            if(value_i > max_value){
                max_value = value_i;
            }
        }

        double delt_value = (max_value - min_value) / 250;
        cv::Point start_point = cv::Point(center.x, center.y + size_one_pannel);
        cv::Point text_place = cv::Point(start_point.x - 50, start_point.y + 20);
        cv::putText(background, vector_i.first, text_place, CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
        cv::putText(background, "Max:"+std::to_string(max_value), cv::Point(text_place.x, text_place.y+20), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
        cv::putText(background, " Min:"+std::to_string(min_value), cv::Point(text_place.x, text_place.y+40), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));

        /// Draw triangles
        for(int i=0; i<num; i++)
        {
            float delt_angle_rad = angle_one_piece * i; /// Note z axis is the opposite
            cv::Point middle_point, left_point, right_point;
            rotateVector(center, start_point, delt_angle_rad,middle_point);
            rotateVector(center, middle_point, angle_one_piece/2.f,left_point);
            rotateVector(center, middle_point, -angle_one_piece/2.f,right_point);

            std::vector<cv::Point> contour;
            contour.push_back(center);
            contour.push_back(left_point);
            contour.push_back(right_point);

            std::vector<std::vector<cv::Point >> contours;
            contours.push_back(contour);

            int color = (int)(250 - (vector_i.second[i] - min_value)/delt_value) + 5;
            cv::Scalar color_to_fill = cv::Scalar(0, 0, color);
            if(vector_i.second[i] == min_value) color_to_fill(0) = 150;

            cv::polylines(background, contours, true, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::fillPoly(background, contours, color_to_fill);
        }
        center.x += center_step;
        if(center.x > cols - center_step/2){
            center.x = center_step/2;
            center.y += center_step;
        }
    }

    cv::imshow("costHeadObjects", background);
    cv::waitKey(2);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "display_costs");
    ros::NodeHandle nh;
    ros::Subscriber cost_head_velocity_sub = nh.subscribe("/head_cost/cost_head_velocity", 1, costHeadVelocityCallback);
    ros::Subscriber cost_head_direction_sub = nh.subscribe("/head_cost/cost_head_direction", 1, costHeadDirectionCallback);
    ros::Subscriber cost_head_objects_sub = nh.subscribe("/head_cost/cost_head_objects", 1, costHeadObjectsCallback);
    ros::Subscriber cost_head_fluctuation_sub = nh.subscribe("/head_cost/cost_head_fluctuation", 1, costHeadFluctuationCallback);
    ros::Subscriber cost_head_update_sub = nh.subscribe("/head_cost/cost_head_update", 1, costHeadUpdateCallback);
    ros::Subscriber cost_head_final_sub = nh.subscribe("/head_cost/cost_head_final", 1, costHeadFinalCallback);

    ros::Timer timer = nh.createTimer(ros::Duration(0.05), displayTimer);

    ros::spin();
}