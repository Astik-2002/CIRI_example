#include "geo_utils.hpp"
#include "firi.hpp"
#include <deque>
#include <memory>
#include <Eigen/Eigen>

#include <algorithm>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <algorithm>
#include <vector>
#include <ctime>
#include <boost/type_index.hpp>
#include <chrono> // Include this for time management
#include <vector>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <thread> // Required for std::this_thread::sleep_for
#include <chrono> // Required for std::chrono::milliseconds
#include <pcl/kdtree/kdtree_flann.h>
#include "voxel_map.hpp"
#include "datatype.h"
#include "trajectory.hpp"
#include "ciri.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include "../data_structure/base/polytope.h"
#include "../data_structure/base/ellipsoid.h"
#include <omp.h>

class ObstacleFree
{
    public:
        pcl::PointCloud<pcl::PointXYZ>::Ptr loadCSVToPointCloud(const std::string &filename);
        pcl::PointCloud<pcl::PointXYZ>::Ptr addDepthDependentPoissonNoise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float noise_scaling_factor = 1.0f);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr convert_pcd_to_rgbd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, float r = 255, float g = 255, float b = 255, float a = 255);
        bool generateObstacleFreePolytopes(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                        const std::vector<Eigen::Vector3d> &points,
                        const Eigen::Vector3d &lowCorner,
                        const Eigen::Vector3d &highCorner,
                        const double &range,
                        std::vector<Eigen::MatrixX4d> &hpolys,
                        const double eps = 1.0e-6); // deprecated
        void findPointCloudLimits(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::Vector3d &minpt, Eigen::Vector3d &maxpt);
        void visualizePolytopePCL(pcl::visualization::PCLVisualizer::Ptr &viewer, const std::vector<Eigen::MatrixX4d> &hPolys, pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, Eigen::MatrixXd &path, const std::vector<NodePtr> nodelist, const Trajectory<5> &traj);
        void pclToVoxelMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            voxel_map::VoxelMap &voxelMap,
                            double dilateRadius);

        void pointCloudInflation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr inflated_pcd);
        void convexCover(const Eigen::MatrixXd &path, 
                        const std::vector<Eigen::Vector3d> &pcd,
                        const Eigen::Vector3d &lowCorner,
                        const Eigen::Vector3d &highCorner,
                        const double &range,
                        const double &progress,
                        std::vector<Eigen::MatrixX4d> &hpolys,
                        std::vector<Eigen::MatrixX3d> &tangentObstacles,
                        const double eps = 1.0e-6);
        
        void convexCoverCIRI(const Eigen::MatrixXd &path, 
                        const std::vector<Eigen::Vector3d> &points,
                        const Eigen::Vector3d &lowCorner,
                        const Eigen::Vector3d &highCorner,
                        const double &range,
                        const double &progress,
                        std::vector<Eigen::MatrixX4d> &hpolys,
                        const Eigen::Vector3d &o,
                        std::vector<geometry_utils::Ellipsoid> &tangent_obs,
                        bool uncertanity,
                        const double eps  = 1.0e-6);

        std::vector<Eigen::MatrixX4d> convexCoverParallel(const Eigen::MatrixXd &path, 
                                        const std::vector<Eigen::Vector3d> &points,
                                        const Eigen::Vector3d &lowCorner,
                                        const Eigen::Vector3d &highCorner,
                                        const double &range,
                                        const double &uav_size,
                                        const Eigen::Vector3d &o,
                                        std::vector<geometry_utils::Ellipsoid> &tangent_obs,
                                        bool uncertanity,
                                        const double eps = 1e-6); 

        inline void shortCut(std::vector<Eigen::MatrixX4d> &hpolys);

        void visualizeCIRI(std::vector<Eigen::MatrixX4d> &hpolys, pcl::visualization::PCLVisualizer::Ptr &viewer, int id = 0);
        void visualizeObs(std::vector<geometry_utils::Ellipsoid> &tangent_obs, pcl::visualization::PCLVisualizer::Ptr &viewer, int id);


};