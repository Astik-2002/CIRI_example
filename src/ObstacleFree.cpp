#include "ObstacleFree.hpp"
#include "corridor_finder.h"
#include "trajectory.hpp"
#include "gcopter.hpp"
#include <typeinfo>
#include <math.h>
#include <omp.h>

#include <vtkOpenGLShaderProperty.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkLogger.h>


pcl::PointCloud<pcl::PointXYZ>::Ptr ObstacleFree::loadCSVToPointCloud(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::string line;
    // Skip the header line if there's any
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x, y, z;

        std::getline(ss, x, ',');
        std::getline(ss, y, ',');
        std::getline(ss, z, ',');

        pcl::PointXYZ point;
        point.x = std::stof(x);
        point.y = std::stof(y);
        point.z = std::stof(z);

        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;  // Unorganized point cloud
    cloud->is_dense = true;

    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ObstacleFree::addDepthDependentPoissonNoise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float noise_scaling_factor) 
{
    // Random number generator for Poisson distribution
    std::default_random_engine generator;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr noisy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    noisy_cloud->points.reserve(cloud->points.size());

    for (const auto &point : cloud->points) {
        pcl::PointXYZ noisy_point = point;

        // Poisson noise: lambda is proportional to depth (x-coordinate)
        float lambda_x = noise_scaling_factor * abs(noisy_point.x);
        float lambda_y = noise_scaling_factor * abs(noisy_point.y);

        float lambda_z = noise_scaling_factor * abs(noisy_point.z);

        std::poisson_distribution<int> poisson_distribution_x(lambda_x);
        std::poisson_distribution<int> poisson_distribution_y(lambda_y);
        std::poisson_distribution<int> poisson_distribution_z(lambda_z);

        // Add noise to the z-coordinate (depth)
        float noise_x = static_cast<float>(poisson_distribution_x(generator)) - lambda_x; // Centered around 0
        float noise_y = static_cast<float>(poisson_distribution_y(generator)) - lambda_y; // Centered around 0
        float noise_z = static_cast<float>(poisson_distribution_z(generator)) - lambda_z; // Centered around 0

        noisy_point.x += noise_x;
        noisy_point.y += noise_y;
        noisy_point.z += noise_z;

        noisy_cloud->points.push_back(noisy_point);
    }

    noisy_cloud->width = noisy_cloud->points.size();
    noisy_cloud->height = 1; // Unorganized point cloud
    noisy_cloud->is_dense = true;

    return noisy_cloud;
}

void ObstacleFree::pointCloudInflation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input, pcl::PointCloud<pcl::PointXYZ>::Ptr inflated_pcd)
{
    for(auto pt: cloud_input->points)
    {
        double x = pt.x;
        double y = pt.y;
        double z = pt.z;

        double distance = sqrt(pow(x,2) + pow(y,2) + pow(z,2));

        // if(distance > clip_distance)
        // {
        //     continue;
        // }
        double sigma_x = 0.001063 + 0.0007278*distance + 0.003949*pow(distance, 2) + 0.022*pow(distance, 3/2);
        double sigma_y = 0.04;
        double sigma_z = sigma_y;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution_x(0.0, sigma_x);
        std::normal_distribution<double> distribution_y(0.0, sigma_y);
        std::normal_distribution<double> distribution_z(0.0, sigma_z);

        double noise_x = distribution_x(generator);
        double noise_y = distribution_y(generator);
        double noise_z = distribution_z(generator);

        double dr1 = sqrt(pow(noise_x,2) + pow(noise_y,2) + pow(noise_z,2)) + 0.25; // constant value added
        double dr2 = dr1/2;
        std::vector<double> dr_vec{dr1};
        for(double dr: dr_vec)
        {
            pcl::PointXYZ p1(x + dr, y - dr, z), p2(x + dr, y, z), p3(x + dr, y + dr, z), p4(x, y - dr, z);
            pcl::PointXYZ p5(x, y + dr, z), p6(x - dr, y - dr, z), p7(x - dr, y, z), p8(x - dr, y + dr, z), p9(x + dr, y - dr, z - dr);
            pcl::PointXYZ p10(x + dr, y, z - dr), p11(x + dr, y + dr, z - dr), p12(x, y - dr, z - dr), p13(x, y + dr, z - dr), p14(x - dr, y - dr, z - dr);
            pcl::PointXYZ p15(x - dr, y, z - dr), p16(x - dr, y + dr, z - dr), p17(x + dr, y - dr, z + dr), p18(x + dr, y, z + dr), p19(x + dr, y + dr, z + dr);
            pcl::PointXYZ p20(x, y - dr, z + dr), p21(x, y + dr, z + dr), p22(x - dr, y - dr, z + dr), p23(x - dr, y, z + dr), p24(x - dr, y + dr, z + dr);
            pcl::PointXYZ p25(x, y, z + dr), p26(x, y, z - dr);
            
            inflated_pcd->points.push_back(p1);
            inflated_pcd->points.push_back(p2);
            inflated_pcd->points.push_back(p3);
            inflated_pcd->points.push_back(p4);
            inflated_pcd->points.push_back(p5);
            inflated_pcd->points.push_back(p6);
            inflated_pcd->points.push_back(p7);
            inflated_pcd->points.push_back(p8);
            inflated_pcd->points.push_back(p9);
            inflated_pcd->points.push_back(p10);
            inflated_pcd->points.push_back(p11);
            inflated_pcd->points.push_back(p12);
            inflated_pcd->points.push_back(p13);
            inflated_pcd->points.push_back(p14);
            inflated_pcd->points.push_back(p15);
            inflated_pcd->points.push_back(p16);
            inflated_pcd->points.push_back(p17);
            inflated_pcd->points.push_back(p18);
            inflated_pcd->points.push_back(p19);
            inflated_pcd->points.push_back(p20);
            inflated_pcd->points.push_back(p21);
            inflated_pcd->points.push_back(p22);
            inflated_pcd->points.push_back(p23);
            inflated_pcd->points.push_back(p24);
            inflated_pcd->points.push_back(p25);
            inflated_pcd->points.push_back(p26);

        }
    }
}

void ObstacleFree::pclToVoxelMap(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, voxel_map::VoxelMap &voxelMap, double dilateRadius)
{
    for (const auto &point : input_cloud->points) {
            // Check for invalid points (NaN or Inf)
            if (std::isnan(point.x) || std::isinf(point.x) ||
                std::isnan(point.y) || std::isinf(point.y) ||
                std::isnan(point.z) || std::isinf(point.z)) {
                continue;
            }

            // Mark voxel as occupied
            voxelMap.setOccupied(Eigen::Vector3d(point.x, point.y, point.z));
        }
        int dilateSteps = std::ceil(dilateRadius / voxelMap.getScale());
        voxelMap.dilate(dilateSteps);
}
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ObstacleFree::convert_pcd_to_rgbd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz, float r, float g, float b, float a)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_rgba(new pcl::PointCloud<pcl::PointXYZRGBA>);

    // Resize the new cloud to match the original
    cloud_rgba->width = cloud_xyz->width;
    cloud_rgba->height = cloud_xyz->height;
    cloud_rgba->is_dense = cloud_xyz->is_dense;
    cloud_rgba->points.resize(cloud_xyz->points.size());

    // Copy the points from cloud_xyz to cloud_rgba
    for (size_t i = 0; i < cloud_xyz->points.size(); ++i) {
        pcl::PointXYZRGBA point_rgba;
        point_rgba.x = cloud_xyz->points[i].x;
        point_rgba.y = cloud_xyz->points[i].y;
        point_rgba.z = cloud_xyz->points[i].z;
        
        // Set the color fields, here we initialize them to a default value
        point_rgba.r = r; // Red channel
        point_rgba.g = g; // Green channel
        point_rgba.b = b; // Blue channel
        point_rgba.a = a; // Alpha channel

        cloud_rgba->points[i] = point_rgba;
    }
    return cloud_rgba;
}
void ObstacleFree::convexCover(const Eigen::MatrixXd &path, 
                        const std::vector<Eigen::Vector3d> &points,
                        const Eigen::Vector3d &lowCorner,
                        const Eigen::Vector3d &highCorner,
                        const double &range,
                        const double &progress,
                        std::vector<Eigen::MatrixX4d> &hpolys,
                        std::vector<Eigen::MatrixX3d> &tangentObstacles,
                        const double eps)
{
    hpolys.clear();
    int n = int(path.rows());
    std::vector<Eigen::Vector3d> temp;
    for(int i=0; i<n; i++)
    {
        temp.emplace_back(path(i, 0), path(i, 1), path(i, 2));
    }
    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;
    
    Eigen::MatrixX4d hp, gap;
    Eigen::MatrixX3d tangent_pcd1, tangent_pcd2;
    Eigen::Vector3d a(path(0,0), path(0,1), path(0,2));
    Eigen::Vector3d b = a;
    std::vector<Eigen::Vector3d> valid_pc;
    std::vector<Eigen::Vector3d> bs;
    valid_pc.reserve(points.size());
    for (int i = 0; i < temp.size();i++)
    {
        
        Eigen::Vector3d path_point = temp[i];
        a = b;
        b = path_point;
        bs.emplace_back(b);

        bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
        bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
        bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
        bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
        bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
        bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

        valid_pc.clear();
        for (const Eigen::Vector3d &p : points)
        {
            if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0)
            {
                valid_pc.emplace_back(p);
            }
        }
        Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());

        firi::firi(bd, pc, a, b, hp, tangent_pcd1);

        if (hpolys.size() != 0)
        {
            const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
            if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
                             ((hpolys.back() * ah).array() > -eps).cast<int>().sum())
            {
                firi::firi(bd, pc, a, a, gap, tangent_pcd2, 1);
                tangentObstacles.emplace_back(tangent_pcd2);
                hpolys.emplace_back(gap);
            }
        }
        tangentObstacles.emplace_back(tangent_pcd1);
        hpolys.emplace_back(hp);
    }

}
/*
void ObstacleFree::convexCoverCIRI(const Eigen::MatrixXd &path, 
                        const std::vector<Eigen::Vector3d> &points,
                        const Eigen::Vector3d &lowCorner,
                        const Eigen::Vector3d &highCorner,
                        const double &range,
                        const double &uav_size,
                        std::vector<Eigen::MatrixX4d> &hpolys,
                        const Eigen::Vector3d &o,
                        bool uncertanity,
                        const double eps)
{
    hpolys.clear();
    int n = int(path.rows());
    std::vector<Eigen::Vector3d> temp;
    for(int i=0; i<n; i++)
    {
        temp.emplace_back(path(i, 0), path(i, 1), path(i, 2));
    }
    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;
    
    Eigen::MatrixX4d hp, gap;
    Eigen::MatrixX3d tangent_pcd1, tangent_pcd2;
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    std::vector<Eigen::Vector3d> valid_pc;
    std::vector<Eigen::Vector3d> bs;
    valid_pc.reserve(points.size());
    super_planner::CIRI ciri;
    ciri.setupParams(uav_size, 10); // Setup CIRI with robot radius and iteration number
    std::cout<<"[ciri debug] temp_size:"<< temp.size()<<std::endl;

    for (int i = 0; i < temp.size();i++)
    {
        std::cout<<"[ciri debug] current pair "<<i<<":"<<i+1<<std::endl;
        if(uncertanity)
        {
            std::cout<<"[ciri debug] stochastic polygons "<<std::endl;
        }
        else
        {
            std::cout<<"[ciri debug] deterministic polygons "<<std::endl;
        }
        
        Eigen::Vector3d path_point = temp[i];
        a = b;
        b = path_point;
        bs.emplace_back(b);

        bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
        bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
        bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
        bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
        bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
        bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

        valid_pc.clear();
        for (const Eigen::Vector3d &p : points)
        {
            if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0)
            {
                valid_pc.emplace_back(p);
            }
        }
        if (valid_pc.empty()) {
            std::cerr << "No valid points found for the current segment." << std::endl;
            // hp = bd;
            // hpolys.emplace_back(hp);
            continue;
        }

        Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());

        if (ciri.convexDecomposition(bd, pc, a, b, o, uncertanity) != super_utils::SUCCESS) {
            std::cerr << "CIRI decomposition failed." << std::endl;
            continue;
        }

        geometry_utils::Polytope optimized_poly;
        ciri.getPolytope(optimized_poly);
        hp = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope

        // if (hpolys.size() != 0)
        // {
        //     const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
        //     if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
        //                      ((hpolys.back() * ah).array() > -eps).cast<int>().sum())
        //     {
        //         if (ciri.convexDecomposition(bd, pc, a, a, o, uncertanity) != super_utils::SUCCESS) {
        //             std::cerr << "CIRI decomposition failed." << std::endl;
        //             continue;
        //         }
        //         ciri.getPolytope(optimized_poly);
        //         gap = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope
        //         hpolys.emplace_back(gap);
        //     }
        // }
        hpolys.emplace_back(hp);
    }
}
*/
void ObstacleFree::convexCoverCIRI(const Eigen::MatrixXd &path, 
    const std::vector<Eigen::Vector3d> &points,
    const Eigen::Vector3d &lowCorner,
    const Eigen::Vector3d &highCorner,
    const double &range,
    const double &uav_size,
    std::vector<Eigen::MatrixX4d> &hpolys,
    const Eigen::Vector3d &o,
    std::vector<geometry_utils::Ellipsoid> &tangent_obs,
    bool uncertanity,
    const double eps)
{
    hpolys.clear();
    int n = int(path.rows());
    if (n < 2) {
        std::cerr << "Path must contain at least two points." << std::endl;
        return;
    }

    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
    bd(0, 0) = 1.0;
    bd(1, 0) = -1.0;
    bd(2, 1) = 1.0;
    bd(3, 1) = -1.0;
    bd(4, 2) = 1.0;
    bd(5, 2) = -1.0;

    Eigen::MatrixX4d hp;
    Eigen::Vector3d a, b;
    std::vector<Eigen::Vector3d> valid_pc;
    valid_pc.reserve(points.size());
    super_planner::CIRI ciri;
    ciri.setupParams(uav_size, 1); // Setup CIRI with robot radius and iteration number

    for (int i = 0; i < n - 1; ++i) 
    {
        std::cout << "[ciri debug] current pair " << i << ":" << i + 1 << std::endl;
        if (uncertanity) 
        {
            std::cout << "[ciri debug] stochastic polygons " << std::endl;
        } 
        else 
        {
            std::cout << "[ciri debug] deterministic polygons " << std::endl;
        }

        a = path.row(i);
        b = path.row(i + 1);

        bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
        bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
        bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
        bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
        bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
        bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

        valid_pc.clear();
        for (const Eigen::Vector3d &p : points) 
        {
            if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0) 
            {
                valid_pc.emplace_back(p);
            }
        }
        if (valid_pc.empty()) 
        {
            std::cerr << "No valid points found for the current segment." << std::endl;
            continue;
        }

        Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());

        if (ciri.convexDecomposition(bd, pc, a, b, o, tangent_obs, uncertanity) != super_utils::SUCCESS) 
        {
            std::cerr << "CIRI decomposition failed." << std::endl;
            continue;
        }

        geometry_utils::Polytope optimized_poly;
        ciri.getPolytope(optimized_poly);
        hp = optimized_poly.GetPlanes(); // Assuming getPlanes() returns the planes of the polytope

        hpolys.emplace_back(hp);
    }
}
/*

void ObstacleFree::convexCoverParallel(const Eigen::MatrixXd &path, 
                                      const std::vector<Eigen::Vector3d> &points,
                                      const Eigen::Vector3d &lowCorner,
                                      const Eigen::Vector3d &highCorner,
                                      const double &range,
                                      const double &progress,
                                      std::vector<Eigen::MatrixX4d> &hpolys,
                                      const double eps) {
    // Clear output container
    hpolys.clear();

    // Extract relevant points from the path
    int n = int(path.rows());
    std::vector<Eigen::Vector3d> temp;
    temp.emplace_back(path(0, 0), path(0, 1), path(0, 2));

    for (int i = 1; i < n-1; i++) {
        if (i % 2 == 0) {
            temp.emplace_back(path(i, 0), path(i, 1), path(i, 2));
        }
    }
    temp.emplace_back(path(n - 1, 0), path(n - 1, 1), path(n - 1, 2));

    // Initialize bounding box constraints
    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
    bd(0, 0) = 1.0;
    bd(1, 0) = -1.0;
    bd(2, 1) = 1.0;
    bd(3, 1) = -1.0;
    bd(4, 2) = 1.0;
    bd(5, 2) = -1.0;

    // Prepare for parallel processing
    int m = int(temp.size());
    std::vector<Eigen::MatrixX4d> local_hpolys(m);

    // OpenMP parallel loop
    #pragma omp parallel for shared(points, temp, bd, local_hpolys, m, range, highCorner, lowCorner) default(none)
    for (int i = 0; i < m; i++) {
        // Define thread-local variables
        Eigen::Vector3d a = temp[i];
        Eigen::Vector3d b = temp[i + 1];
        std::vector<Eigen::Vector3d> valid_pc;
        // std::cout<<"[parallel debug] point size "<<points.size()<<std::endl;
        // Update bounding box constraints for the current segment
        bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
        bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
        bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
        bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
        bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
        bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

        // Filter valid points within the bounding box
        for (const Eigen::Vector3d &p : points) {
            if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0) {
                valid_pc.emplace_back(p);
            }
        }

        // Perform convex hull calculation for the valid points
        Eigen::MatrixX4d hp;
        if (!valid_pc.empty()) {
            Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());
            firi::firi(bd, pc, a, b, hp);
        }

        // Store the result in the thread-local container
        local_hpolys[i] = hp;
    }

    // Combine results from all threads into the output container
    hpolys.resize(m);
    for (int i = 0; i < m; i++) {
        hpolys[i] = local_hpolys[i];
    }
    std::cout<<"[parallel convex decomp debug] size of hpolys: "<<hpolys.size()<<std::endl;

}
*/

void ObstacleFree::findPointCloudLimits(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, Eigen::Vector3d &min_pt, Eigen::Vector3d &max_pt)
{
    if (cloud_in->empty())
    {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    // Initialize min and max points with the first point in the cloud
    min_pt = Eigen::Vector3d(cloud_in->points[0].x, cloud_in->points[0].y, cloud_in->points[0].z);
    max_pt = Eigen::Vector3d(cloud_in->points[0].x, cloud_in->points[0].y, cloud_in->points[0].z);

    // Iterate through the point cloud to find min and max coordinates
    for (const auto& point : cloud_in->points)
    {
        // Update minimum coordinates
        if (point.x < min_pt(0)) min_pt(0) = point.x;
        if (point.y < min_pt(1)) min_pt(1) = point.y;
        if (point.z < min_pt(2)) min_pt(2) = point.z;

        // Update maximum coordinates
        if (point.x > max_pt(0)) max_pt(0) = point.x;
        if (point.y > max_pt(1)) max_pt(1) = point.y;
        if (point.z > max_pt(2)) max_pt(2) = point.z;
    }

    std::cout << "Point cloud limits:" << std::endl;
    std::cout << "Min: (" << min_pt(0) << ", " << min_pt(1) << ", " << min_pt(2) << ")" << std::endl;
    std::cout << "Max: (" << max_pt(0) << ", " << max_pt(1) << ", " << max_pt(2) << ")" << std::endl;
}


void ObstacleFree::visualizePolytopePCL(pcl::visualization::PCLVisualizer::Ptr &viewer, const std::vector<Eigen::MatrixX4d> &hPolys, pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, Eigen::MatrixXd &Path_rrt, const std::vector<NodePtr> nodelist, const Trajectory<5> &traj) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<pcl::Vertices> polygons;

    for (const auto &hPoly : hPolys) {
        Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
        geo_utils::enumerateVs(hPoly, vPoly);

        quickhull::QuickHull<double> tinyQH;
        const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
        const auto &idxBuffer = polyHull.getIndexBuffer();

        // Add vertices to PCL PointCloud
        int baseIndex = cloud->size();
        for (int i = 0; i < vPoly.cols(); ++i) {
            cloud->emplace_back(vPoly(0, i), vPoly(1, i), vPoly(2, i));
        }

        // Add triangles to PCL PolygonMesh
        for (size_t i = 0; i < idxBuffer.size(); i += 3) {
            pcl::Vertices triangle;
            triangle.vertices.push_back(baseIndex + idxBuffer[i]);
            triangle.vertices.push_back(baseIndex + idxBuffer[i + 1]);
            triangle.vertices.push_back(baseIndex + idxBuffer[i + 2]);
            polygons.push_back(triangle);
        }
    }

    // Convert to PolygonMesh
    pcl::PolygonMesh polyMesh;
    pcl::toPCLPointCloud2(*cloud, polyMesh.cloud);
    polyMesh.polygons = polygons;
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr path_rgba(new pcl::PointCloud<pcl::PointXYZRGBA>);
        for(int i=0; i<int(Path_rrt.rows()); i++)
        {
            pcl::PointXYZRGBA point_rgba;
            point_rgba.x = Path_rrt(i, 0);
            point_rgba.y = Path_rrt(i, 1);
            point_rgba.z = Path_rrt(i, 2);
            
            // Set the color fields, here we initialize them to a default value
            point_rgba.r = 0; // Red channel
            point_rgba.g = 0; // Green channel
            point_rgba.b = 255; // Blue channel
            point_rgba.a = 255; // Alpha channel
            // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
            // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres

            path_rgba->points.push_back(point_rgba);
        }
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr node_rgba(new pcl::PointCloud<pcl::PointXYZRGBA>);
    for( int i=0; i<nodelist.size(); i++)
    {
        pcl::PointXYZRGBA point_rgba;
            point_rgba.x = nodelist[i]->coord[0];
            point_rgba.y = nodelist[i]->coord[1];
            point_rgba.z = nodelist[i]->coord[2];
            
            // Set the color fields, here we initialize them to a default value
            point_rgba.r = 255; // Red channel
            point_rgba.g = 0; // Green channel
            point_rgba.b = 0; // Blue channel
            point_rgba.a = 255; // Alpha channel
            // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
            // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres

            node_rgba->points.push_back(point_rgba);
    }


    // Trajectory visualization
    
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj_points(new pcl::PointCloud<pcl::PointXYZRGBA>());
    double T = 0.01; // Sampling interval
    Eigen::Vector3d lastX = traj.getPos(0.0);

    for (double t = T; t < traj.getTotalDuration(); t += T) {
        Eigen::Vector3d X = traj.getPos(t);

        // Add the current point to the trajectory point cloud
        pcl::PointXYZRGBA point;
        point.x = X(0);
        point.y = X(1);
        point.z = X(2);
        point.r = 0;
        point.g = 255;
        point.b = 0;
        point.a = 255;
        traj_points->points.push_back(point);
    }

    // Visualize using PCL
    viewer->addPolygonMesh(polyMesh, "polytope_mesh");
    viewer->addPointCloud(input_cloud, "pcd");
    viewer->addPointCloud(path_rgba, "rrt path");
    viewer->addPointCloud(traj_points, "trajectory_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.7, "pcd");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "pcd");

    // viewer->addPointCloud(node_rgba,"nodelist");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "rrt path");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "pcd");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "trajectory_points");
}

inline void ObstacleFree::shortCut(std::vector<Eigen::MatrixX4d> &hpolys)
{
    std::vector<Eigen::MatrixX4d> htemp = hpolys;
    if (htemp.size() == 1)
        {
            Eigen::MatrixX4d headPoly = htemp.front();
            htemp.insert(htemp.begin(), headPoly);
        }
    hpolys.clear();

    int M = htemp.size();
    Eigen::MatrixX4d hPoly;
    bool overlap;
    std::deque<int> idices;
    idices.push_front(M - 1);
    for (int i = M - 1; i >= 0; i--)
    {
        for (int j = 0; j < i; j++)
        {
            if (j < i - 1)
            {
                overlap = geo_utils::overlap(htemp[i], htemp[j], 0.01);
            }
            else
            {
                overlap = true;
            }
            if (overlap)
            {
                idices.push_front(j);
                i = j + 1;
                break;
            }
        }
    }
    for (const auto &ele : idices)
    {
        hpolys.push_back(htemp[ele]);
    }
}

void ObstacleFree::visualizeCIRI(std::vector<Eigen::MatrixX4d> &hpolys, pcl::visualization::PCLVisualizer::Ptr &viewer, int id)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<pcl::Vertices> polygons;

    int r = ((id + 1) * 77) % 256;
    int g = ((id + 1) * 137) % 256;
    int b = ((id + 1) * 199) % 256;

    for (const auto &hPoly : hpolys) 
    {
        Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
        geo_utils::enumerateVs(hPoly, vPoly);

        quickhull::QuickHull<double> tinyQH;
        const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
        const auto &idxBuffer = polyHull.getIndexBuffer();

        // Add vertices to PCL PointCloud with RGB color
        int baseIndex = cloud->size();
        for (int i = 0; i < vPoly.cols(); ++i) {
            pcl::PointXYZRGB point;
            point.x = vPoly(0, i);
            point.y = vPoly(1, i);
            point.z = vPoly(2, i);
            // Assign a color to the point (e.g., red)
            point.r = r;
            point.g = g;
            point.b = b;
            cloud->push_back(point);
        }

        // Add triangles to PCL PolygonMesh
        for (size_t i = 0; i < idxBuffer.size(); i += 3) {
            pcl::Vertices triangle;
            triangle.vertices.push_back(baseIndex + idxBuffer[i]);
            triangle.vertices.push_back(baseIndex + idxBuffer[i + 1]);
            triangle.vertices.push_back(baseIndex + idxBuffer[i + 2]);
            polygons.push_back(triangle);
        }
    }

    // Convert to PolygonMesh
    pcl::PolygonMesh polyMesh;
    pcl::toPCLPointCloud2(*cloud, polyMesh.cloud);
    polyMesh.polygons = polygons;
    viewer->addPolygonMesh(polyMesh, "polytope_mesh_"+std::to_string(id));
}

// void ObstacleFree::visualizeObs(std::vector<geometry_utils::Ellipsoid> &tangent_obs, pcl::visualization::PCLVisualizer::Ptr &viewer, int id)
// {
//     double r = ((id + 1) * 77) % 256 / 255.0;
//     double g = ((id + 1) * 137) % 256 / 255.0;
//     double b = ((id + 1) * 199) % 256 / 255.0;

//     for (int i = 0; i < tangent_obs.size(); i++)
//     {
//         geometry_utils::Ellipsoid obs = tangent_obs[i];
//         auto center = obs.d();  // Ellipsoid center
//         auto shape_matrix = obs.C();  // Shape matrix (assumed to be covariance-like)

//         // Compute SVD decomposition
//         // Eigen::JacobiSVD<Eigen::Matrix3d> svd(shape_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
//         Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::FullPivHouseholderQRPreconditioner> svd(shape_matrix, Eigen::ComputeFullU);
//         Eigen::Matrix3d U = svd.matrixU();
//         Eigen::Vector3d radii = svd.singularValues().cwiseSqrt();
//         // Create transformation matrix
//         Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
//         transform.translation() << center[0], center[1], center[2];
//         transform.linear() = U;  // Apply rotation from singular vectors
//         std::string ellipsoid_id = "ellipsoid_" + std::to_string(i) + "_" + std::to_string(id);

//         // Add ellipsoid with correct radii and orientation
//         viewer->addEllipsoid(transform, radii[0], radii[1], radii[2], ellipsoid_id);
//         viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, ellipsoid_id);
//         viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, ellipsoid_id);
//     }
// }

void ObstacleFree::visualizeObs(std::vector<geometry_utils::Ellipsoid> &tangent_obs, pcl::visualization::PCLVisualizer::Ptr &viewer, int id)
{
    double r = ((id + 1) * 77) % 256/255.0;
    double g = ((id + 1) * 137) % 256/255.0;
    double b = ((id + 1) * 199) % 256/255.0;

    for (int i = 0; i < tangent_obs.size(); i++)
    {
        geometry_utils::Ellipsoid obs = tangent_obs[i];
        auto center = obs.d();
        auto axes = obs.r();
        auto Rot = obs.R();
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.linear() = Rot;
        transform.translation() << center[0], center[1], center[2];

        std::string ellipsoid_id = "ellipsoid_" + std::to_string(i) + "_" + std::to_string(id);

        viewer->addEllipsoid(transform, axes[0], axes[1], axes[2],
                            ellipsoid_id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, ellipsoid_id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, ellipsoid_id);

    }
}

int main(int argc, char** argv)
{
    vtkObject::GlobalWarningDisplayOff();

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
        return -1;
    }
    std::cout<<"printing for debugging1"<<std::endl;
    ObstacleFree sfc_generator;
    Trajectory<5> traj;

    safeRegionRrtStar rrt_path_gen;
    // rrt gen parameters
    float _uav_size = 0.4;
    float _safety_margin = 1.5*_uav_size;
    float _search_margin = 0.5;
    float _max_radius = 5.0;
    float _sensing_range = 25.0;

    float x_l = 0.0;
    float x_h = 12.0;
    float y_l = -7.0;
    float y_h = 7.0;
    float z_l = 0.0;
    float z_l2 = 0.0;
    float z_h = 5.5;
    float z_h2 = 5.5;
    float local_range = 2.0, sample_portion=0.25, goal_portion=0.05;
    int max_iter=100000;
    std::string csv_file = argv[1];
    float voxelWidth = 0.25;
    float dilateRadius = 0.50;
    float leafsize = 0.50;
    const Eigen::Vector3i xyz((x_h - x_l) / voxelWidth,
                                  (y_h - y_l) / voxelWidth,
                                  (z_h2 - z_l2) / voxelWidth);

    const Eigen::Vector3d offset(x_l, y_l, z_l2);

    voxel_map::VoxelMap V_map(xyz, offset, voxelWidth);
    auto input_cloud = sfc_generator.loadCSVToPointCloud(csv_file);
    std::cout<<"printing for debugging2"<<std::endl;
    if (!input_cloud) {
        std::cerr << "Failed to load point cloud from CSV." << std::endl;
        return -1;
    }
    

    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d a(0.0, 0.0, 1.0);
    Eigen::Vector3d b(30.0, -3.0, 1.0);

    sfc_generator.pclToVoxelMap(input_cloud, V_map, dilateRadius );
    std::vector<Eigen::Vector3d> eigen_points;

    auto time_bef_voxel_gen = std::chrono::steady_clock::now();
    V_map.getSurf(eigen_points);
    auto time_aft_voxel_gen = std::chrono::steady_clock::now();
    auto elapsed_voxel = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_voxel_gen - time_bef_voxel_gen).count()*0.001;
    std::cout<<"[voxel comparision] time taken in voxel dilation: "<<elapsed_voxel<<std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr V_map_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for(auto point : eigen_points)
    {
        pcl::PointXYZ pcd_point;
        pcd_point.x = point(0);
        pcd_point.y = point(1);
        pcd_point.z = point(2);
        V_map_cloud->points.push_back(pcd_point);

    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr inflated_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    auto time_bef_voxel_inf = std::chrono::steady_clock::now();

    sfc_generator.pointCloudInflation(input_cloud, inflated_cloud);
    
    auto time_aft_voxel_inf = std::chrono::steady_clock::now();
    auto elapsed_inf = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_voxel_inf - time_bef_voxel_inf).count()*0.001;
    std::cout<<"[voxel comparision] time taken in new inflation: "<<elapsed_inf<<std::endl;
    std::cout<<"input pointcloud size: "<<input_cloud->points.size()<<std::endl;
    std::cout<<"dilated pcd size: "<<inflated_cloud->points.size()<<std::endl;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(leafsize, leafsize, leafsize); // Set the voxel grid size (adjust as needed)
    voxel_filter.filter(*filtered_cloud);

    std::cout<<"input pointcloud size: "<<input_cloud->points.size()<<std::endl;
    std::cout<<"dilated pcd size: "<<inflated_cloud->points.size()<<std::endl;
    std::cout<<"filtered pcd: "<<filtered_cloud->points.size()<<std::endl;

    // Show the point cloud
    auto t_rrt_start = std::chrono::steady_clock::now();
    rrt_path_gen.setInput(*input_cloud, origin);
    rrt_path_gen.setParam(_safety_margin, _search_margin, _max_radius, _sensing_range);
    rrt_path_gen.setStartPt(a, b);
    rrt_path_gen.setPt(a, b, x_l, x_h, y_l, y_h, z_l, z_h, _sensing_range, max_iter, sample_portion, goal_portion);
    rrt_path_gen.SafeRegionExpansion(0.1);
    auto t_rrt_end = std::chrono::steady_clock::now();
    auto elapsed_rrt = std::chrono::duration_cast<std::chrono::milliseconds>(t_rrt_end - t_rrt_start).count()*0.001;
    std::cout<<"[time comp] RRT Time taken: "<<elapsed_rrt<<std::endl;

    auto path_radius_pair = rrt_path_gen.getPath();
    bool path_exist = rrt_path_gen.getPathExistStatus();

/*
    auto t_rrt_mmd_start = std::chrono::steady_clock::now();
    rrt_path_gen_mmd.setInput(*input_cloud);
    rrt_path_gen_mmd.setParam(_safety_margin, _search_margin, _max_radius, _sensing_range);
    rrt_path_gen_mmd.setStartPt(a, b);
    rrt_path_gen_mmd.setPt(a, b, x_l, x_h, y_l, y_h, z_l, z_h, _sensing_range, max_iter, sample_portion, goal_portion);
    rrt_path_gen_mmd.SafeRegionExpansion(150.0);
    auto t_rrt_mmd_end = std::chrono::steady_clock::now();
    auto elapsed_mmd = std::chrono::duration_cast<std::chrono::milliseconds>(t_rrt_mmd_end - t_rrt_mmd_start).count()*0.001;
    std::cout<<"[time comp] RRT Time taken with MMD: "<<elapsed_mmd<<std::endl;

    auto t_rrt_gaus_start = std::chrono::steady_clock::now();
    rrt_path_gen_gaus.setInput(*input_cloud);
    rrt_path_gen_gaus.setParam(_safety_margin, _search_margin, _max_radius, _sensing_range);
    rrt_path_gen_gaus.setStartPt(a, b);
    rrt_path_gen_gaus.setPt(a, b, x_l, x_h, y_l, y_h, z_l, z_h, _sensing_range, max_iter, sample_portion, goal_portion);
    rrt_path_gen_gaus.SafeRegionExpansion(150.0);
    auto t_rrt_gaus_end = std::chrono::steady_clock::now();
    auto elapsed_gaus = std::chrono::duration_cast<std::chrono::milliseconds>(t_rrt_gaus_end - t_rrt_gaus_start).count()*0.001;
    std::cout<<"[time comp] RRT Time taken with gaussian: "<<elapsed_gaus<<std::endl;

    auto path_radius_pair = rrt_path_gen.getPath();
    auto path_rrt1 = rrt_path_gen.getPath().first;
    bool path_exist = rrt_path_gen.getPathExistStatus();

    auto path_rrt_mmd = rrt_path_gen_mmd.getPath().first;
    bool path_exist_mmd = rrt_path_gen_mmd.getPathExistStatus();

    auto path_rrt_gaus = rrt_path_gen_gaus.getPath().first;
    bool path_exist_gaus = rrt_path_gen_gaus.getPathExistStatus();

    if(path_exist && path_exist_mmd && path_exist_gaus)
    {   

        pcl::visualization::PCLVisualizer::Ptr viewer_temp(new pcl::visualization::PCLVisualizer("Point cloud visualization"));
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr path_rrt_og(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr path_mmd(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr path_gaus(new pcl::PointCloud<pcl::PointXYZRGBA>);

        for(int i=0; i<int(path_rrt1.rows()); i++)
        {
            pcl::PointXYZRGBA point_rgba;
            point_rgba.x = path_rrt1(i,0);
            point_rgba.y = path_rrt1(i,1);
            point_rgba.z = path_rrt1(i,2);
            
            // Set the color fields, here we initialize them to a default value
            point_rgba.r = 255; // Red channel
            point_rgba.g = 0; // Green channel
            point_rgba.b = 0; // Blue channel
            point_rgba.a = 100; // Alpha channel
            // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
            // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres
            path_rrt_og->points.push_back(point_rgba);
        }

        for(int i=0; i<int(path_rrt_mmd.rows()); i++)
        {
            pcl::PointXYZRGBA point_rgba;
            point_rgba.x = path_rrt_mmd(i,0);
            point_rgba.y = path_rrt_mmd(i,1);
            point_rgba.z = path_rrt_mmd(i,2);
            
            // Set the color fields, here we initialize them to a default value
            point_rgba.r = 0; // Red channel
            point_rgba.g = 255; // Green channel
            point_rgba.b = 0; // Blue channel
            point_rgba.a = 100; // Alpha channel
            // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
            // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres
            path_mmd->points.push_back(point_rgba);
        }

        for(int i=0; i<int(path_rrt_gaus.rows()); i++)
        {
            pcl::PointXYZRGBA point_rgba;
            point_rgba.x = path_rrt_gaus(i,0);
            point_rgba.y = path_rrt_gaus(i,1);
            point_rgba.z = path_rrt_gaus(i,2);
            
            // Set the color fields, here we initialize them to a default value
            point_rgba.r = 0; // Red channel
            point_rgba.g = 0; // Green channel
            point_rgba.b = 255; // Blue channel
            point_rgba.a = 100; // Alpha channel
            // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
            // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres
            path_gaus->points.push_back(point_rgba);
        }

        viewer_temp->addPointCloud(path_rrt_og, "rrt_og");
        viewer_temp->addPointCloud(path_mmd, "rrt_mmd");
        viewer_temp->addPointCloud(path_gaus, "rrt_gaus");

        viewer_temp->addPointCloud(input_cloud, "pcd");
        viewer_temp->setBackgroundColor(0.0, 0.0, 0.0);
        viewer_temp->addCoordinateSystem(1.0);
        viewer_temp->setRepresentationToWireframeForAllActors();
        viewer_temp->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "pcd");
        viewer_temp->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "rrt_og");
        viewer_temp->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "rrt_mmd");
        viewer_temp->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "rrt_gaus");

        while (!viewer_temp->wasStopped()) {
            viewer_temp->spinOnce(100);
        }

    }
*/
    

    bool ciri_testing = true;
    if(ciri_testing)
    {
        _uav_size = 0.2;

        std::mt19937 gen;
        std::uniform_real_distribution<> dis_x(2.0, 8.0);
        std::uniform_real_distribution<> dis_y(-2, 2);
        std::uniform_real_distribution<> dis_z(0.5, 2);
        std::mt19937 gen2(23);
        a[0] = 3.5;
        a[1] = 0.0;
        a[2] = 1.0;

        b[0] = 4.5;
        b[1] = 0.0;
        b[2] = 1.0;

        Eigen::Vector3d min_pt{-9.0, -9.0, -9.0};
        Eigen::Vector3d max_pt{9.0, 9.0, 9.0};
        Eigen::Vector3d o = min_pt;
        pcl::PointCloud<pcl::PointXYZ>::Ptr demo_input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<Eigen::Vector3d> demo_points;
        std::vector<Eigen::Vector3d> noise_vec;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr seed_pt_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointXYZRGB a_pt;
        a_pt.x = a[0];
        a_pt.y = a[1];
        a_pt.z = a[2];
        a_pt.r = 0;
        a_pt.g = 255;
        a_pt.b = 0;

        pcl::PointXYZRGB b_pt;
        b_pt.x = b[0];
        b_pt.y = b[1];
        b_pt.z = b[2];
        b_pt.r = 0;
        b_pt.g = 255;
        b_pt.b = 0;
        
        seed_pt_cloud->points.push_back(a_pt);
        seed_pt_cloud->points.push_back(b_pt);
        
        int num_points = 50;
        for(int i=0; i<num_points; i++)
        {
            double x = dis_x(gen);
            double y = dis_y(gen);
            double z = dis_z(gen);

            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = z;
            demo_input_cloud->points.push_back(point);

            Eigen::Vector3d eigen_point(x,y,z);
            demo_points.push_back(eigen_point);

            double sigma_x = (0.001063 + 0.0007278 * x + 0.003949 * x * x)*sqrt(11.33) + _uav_size;
            double sigma_y = 0.04*sqrt(11.33) + _uav_size;
            double sigma_z = 0.04*sqrt(11.33) + _uav_size;
            Eigen::Vector3d noise_std{sigma_x, sigma_y, sigma_z};
            // std::cout<<"[obstacle debug] obstacle point: "<<eigen_point.transpose()<<std::endl;
            // std::cout<<"[obstacle debug] noise std: "<<noise_std.transpose()<<std::endl;
            noise_vec.push_back(noise_std);
        }

        std::vector<Eigen::MatrixX4d> CIRI_hpolys;
        std::vector<Eigen::MatrixX4d> CIRI_hpolys2;

        Eigen::MatrixXd path_rrt(2, 3);
        path_rrt.row(0) = a.transpose();
        path_rrt.row(1) = b.transpose();
        std::vector<geometry_utils::Ellipsoid> tangent_obs;
        std::vector<geometry_utils::Ellipsoid> tangent_obs2;
        
        sfc_generator.convexCoverCIRI(path_rrt, demo_points, min_pt, max_pt, 1.5, _uav_size, CIRI_hpolys, o, tangent_obs, true);
        // sfc_generator.convexCoverCIRI(path_rrt, demo_points, min_pt, max_pt, 1.0, _uav_size, CIRI_hpolys2, o, tangent_obs2, false);

        std::cout<<"CIRI_hpolys[0]: "<<CIRI_hpolys[0]<<std::endl;
        
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Polytope Visualization"));
        int id = 0;
        sfc_generator.visualizeCIRI(CIRI_hpolys, viewer, id);
        // sfc_generator.visualizeCIRI(CIRI_hpolys2, viewer, ++id);
        id = 0;
        sfc_generator.visualizeObs(tangent_obs, viewer, id);
        // sfc_generator.visualizeObs(tangent_obs2, viewer, ++id);
        viewer->addPointCloud(demo_input_cloud, "input_pcd");
        viewer->addPointCloud(seed_pt_cloud, "seed_pcd");

        for (size_t i = 0; i < demo_points.size(); ++i) {
            const Eigen::Vector3d& center = demo_points[i];
            const Eigen::Vector3d& axes = noise_vec[i];
            pcl::PointXYZ point_rgba;
            point_rgba.x = demo_points[i][0];
            point_rgba.y = demo_points[i][1];
            point_rgba.z = demo_points[i][2];

            // Create an Isometry3d transformation matrix for the ellipsoid's position
            // viewer->addSphere(point_rgba, _uav_size, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres

        }
        
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "seed_pcd");

        viewer->setBackgroundColor(0.0, 0.0, 0.0);
        viewer->addCoordinateSystem(1.0);
        viewer->setRepresentationToWireframeForAllActors();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }
    }
    else
    {
        if(path_exist)
    {
        Eigen::MatrixXd path_rrt = path_radius_pair.first;
        // Eigen::MatrixXd path_rrt_mmd = path_radius_pair_mmd.first;

        Eigen::MatrixXd path_skeleton = path_rrt;
        Eigen::VectorXd radius_rrt = path_radius_pair.second;
        // Eigen::VectorXd radius_rrt_mmd = path_radius_pair_mmd.second;
        Eigen::VectorXd radius_skeleton = radius_rrt;
        Eigen::MatrixXd corridor_skeleton;
        int num_points = path_rrt.rows();
        std::random_device rd;  // Seed for the random number engine
        std::mt19937 gen(rd()); // Mersenne Twister random number generator
        double rho1 = 0.99;
        double rho2 = 1-rho1;
        
        /*
        for(int i=1; i<num_points; i++)
        {
            Eigen::Vector3d path_point(path_skeleton(i,0), path_skeleton(i,1), path_skeleton(i,2));
            Eigen::Vector3d anchor_point(path_skeleton(i-1,0), path_skeleton(i-1,1), path_skeleton(i-1,2));
            double r1 = radius_skeleton(i-1);
            double r2 = radius_skeleton(i);
            double d = (path_point - anchor_point).norm();
            double phi_x = 2*r2;
            double phi_y = 50*phi_x;
            double phi_z = 50*phi_x;
            Eigen::Vector3d best_radius_vec;
            double h1 = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
            double rc = sqrt(r1 * r1 - h1 * h1);
            double best_score = rho1*r2 + rho2*rc;
            std::normal_distribution<> dist_x(anchor_point[0], sqrt(phi_x));
            std::normal_distribution<> dist_y(anchor_point[1], sqrt(phi_y));
            std::normal_distribution<> dist_z(anchor_point[0], sqrt(phi_z));

            for(int j=0; j < 1000; j++)
            {
                Eigen::Vector3d sample(dist_x(gen), dist_x(gen), dist_x(gen));
                
                double r3 = rrt_path_gen.radiusSearch(sample);
                double dn = (sample - anchor_point).norm();

                double hn = (r1 * r1 - r3 * r3 + dn * dn) / (2 * dn);
                double rcn = sqrt(r1 * r1 - hn * hn);
                double score = rho1*r3 + rho2*rcn;
                if(r3 < _max_radius) continue;
                if(score > best_score)
                {
                    best_score = score;
                    path_skeleton(i,0) = sample[0];
                    path_skeleton(i,1) = sample[1];
                    path_skeleton(i,2) = sample[2];
                    std::cout<<"score updated for "<<i<<" old radius: "<<radius_skeleton[i]<<" new radius "<<r3<<std::endl;

                    radius_skeleton[i] = r3;
                }
            }

        }
*/


        std::vector<Eigen::MatrixX4d> hpolys;
        std::vector<Eigen::MatrixX4d> hpolys_parallel;
        std::vector<Eigen::MatrixX3d> tangentObstacles;
        Eigen::Vector3d min_pt(x_l, y_l, z_l);
        Eigen::Vector3d max_pt(x_h, y_h, z_h);
        std::vector<NodePtr> nodelist = rrt_path_gen.getTree();
        std::cout<<"nodelist size: "<<nodelist.size()<<std::endl;
        float progress = _max_radius;
        std::vector<Eigen::Vector3d> eigen_points;

        for (const auto& point : filtered_cloud->points)
        {
            Eigen::Vector3d eigen_point(point.x, point.y, point.z);
            eigen_points.push_back(eigen_point);
        }
        // V_map.getSurf(eigen_points);
        std::vector<Eigen::MatrixX4d> CIRI_hpolys;
        std::vector<Eigen::MatrixX4d> CIRI_hpolys_deterministic;
        auto time_bef_firi = std::chrono::steady_clock::now();
        std::vector<geometry_utils::Ellipsoid> tangent_obs;
        sfc_generator.convexCover(path_rrt, eigen_points, min_pt, max_pt, 1.0, 1.0, hpolys, tangentObstacles);
        auto time_aft_firi = std::chrono::steady_clock::now();

        auto elapsed_firi = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_firi - time_bef_firi).count()*0.001;

        Eigen::Vector3d o{0.0, 0.0, 0.0};
        auto time_bef_ciri = std::chrono::steady_clock::now();

        sfc_generator.convexCoverCIRI(path_rrt, eigen_points, min_pt, max_pt, 1.5, _uav_size, CIRI_hpolys, o, tangent_obs, true);
        auto time_aft_ciri = std::chrono::steady_clock::now();

        auto elapsed_ciri = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_ciri - time_bef_ciri).count()*0.001;

        // sfc_generator.convexCoverCIRI(path_rrt, eigen_points, min_pt, max_pt, 1.5, _uav_size, CIRI_hpolys_deterministic, o, tangent_obs, false);
        if(hpolys.size() != tangentObstacles.size())
        {
            std::cout<<"[mmd debug] something is wrong"<<hpolys.size()<<" : "<<tangentObstacles.size()<<std::endl;   
        }
        else
        {
            for(int i=0; i<hpolys.size(); i++)
            {
                if(hpolys[i].rows() != tangentObstacles[i].rows()) std::cout<<"[mmd debug] if2 something is wrong"<<std::endl;
            }
        }
        // sfc_generator.shortCut(hpolys);
        
        // auto time_bef_firi_parallel = std::chrono::steady_clock::now();
        // sfc_generator.convexCoverParallel(path_rrt, eigen_points, min_pt, max_pt, 1.0, 1.0, hpolys_parallel);
        // // sfc_generator.shortCut(hpolys_parallel);
        auto time_aft_firi_parallel = std::chrono::steady_clock::now();
        // auto elapsed_firi_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_firi_parallel - time_bef_firi_parallel).count()*0.001;

        gcopter::GCOPTER_PolytopeSFC gCopter;
        Eigen::Vector3d front = a;
        int n = path_skeleton.rows();

        Eigen::Vector3d back = b;
        std::vector<Eigen::MatrixX3d> t_obs = tangentObstacles;
        std::vector<std::vector<float>> trust_vector;
        for(int i = 0; i<tangentObstacles.size(); i++)
        {
            std::vector<float> temp_vec;
            for(int j=0; j<tangentObstacles[i].rows(); j++)
            {
                if(tangentObstacles[i](j, 0) != INFINITY && tangentObstacles[i](j, 1) != INFINITY && tangentObstacles[i](j, 2) != INFINITY )
                {
                    double distance = sqrt(pow(tangentObstacles[i](j, 0), 2) + pow(tangentObstacles[i](j, 1), 2) + pow(tangentObstacles[i](j, 2), 2));
                    double sigma = 0.001063 + 0.0007278 * distance + 
                   0.003949 * distance * distance + 
                   0.022 * pow(distance, 1.5);

                    tangentObstacles[i](j, 0) = 0.04;
                    tangentObstacles[i](j, 1) = 0.04;
                    tangentObstacles[i](j, 2) = sigma;

                    Eigen::Vector3d tangentRow = tangentObstacles[i].row(j);
                    Eigen::Vector4d hpolyRow = hpolys[i].row(j); // 4D vector

                    // Compute dot product using only the first 3 elements of hpolyRow
                    double dot_product = tangentRow.dot(hpolyRow.head<3>());
                                    double norm_tangent = tangentRow.norm();
                    double norm_hpoly = hpolyRow.head<3>().norm();

                    // Compute normalized dot product (cosine similarity)
                    double normalized_dot_product = 0.0;
                    if (norm_tangent > 1e-6 && norm_hpoly > 1e-6) // Avoid division by zero
                    {
                        normalized_dot_product = abs(dot_product / (norm_tangent * norm_hpoly));
                    }
                    temp_vec.push_back((normalized_dot_product + 1)*_safety_margin);
                }
                else
                {
                    temp_vec.push_back(_safety_margin);
                }
            }
            trust_vector.push_back(temp_vec);
        }
        // GCopter parameters
        Eigen::Matrix3d iniState;
        Eigen::Matrix3d finState;
        iniState << front, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        finState << back, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        Eigen::VectorXd magnitudeBounds(5);
        Eigen::VectorXd penaltyWeights(5);
        Eigen::VectorXd physicalParams(6);
        std::vector<float> chiVec = {10000, 10000, 10000, 10000, 100000};
        magnitudeBounds(0) = 4.0;
        magnitudeBounds(1) = 2.1;
        magnitudeBounds(2) = 1.05;
        magnitudeBounds(3) = 2.0;
        magnitudeBounds(4) = 12.0;
        penaltyWeights(0) = chiVec[0];
        penaltyWeights(1) = chiVec[1];
        penaltyWeights(2) = chiVec[2];
        penaltyWeights(3) = chiVec[3];
        penaltyWeights(4) = chiVec[4];
        physicalParams(0) = 0.61;
        physicalParams(1) = 9.8;
        physicalParams(2) = 0.70;
        physicalParams(3) = 0.80;
        physicalParams(4) = 0.01;
        physicalParams(5) = 0.0001;
        int quadratureRes = 16;
        float weightT = 20.0;
        float smoothingEps = 0.01;
        float relcostto1 = 1e-3;
        traj.clear();
        std::vector<Eigen::Vector3d> rrt_vec;
        for(int i = 0; i<path_rrt.rows(); i++)
        {
            Eigen::Vector3d pt(path_rrt(i, 0), path_rrt(i, 1), path_rrt(i, 2));
            rrt_vec.push_back(pt);
        }
        auto time_bef_gcopter = std::chrono::steady_clock::now();

        if (!gCopter.setup(weightT, iniState, finState, hpolys, rrt_vec, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams, trust_vector))
        {
            std::cout<<"gcopter returned false during setup"<<std::endl;
        }
        if (std::isinf(gCopter.optimize(traj, relcostto1)))
        {
            std::cout<<"gcopter optimization cost is infinity"<<std::endl;
        }
        auto time_aft_gcopter = std::chrono::steady_clock::now();

        auto elapsed_gcopter = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_gcopter - time_bef_gcopter).count()*0.001;
        std::cout<<" time in firi (in seconds): "<<elapsed_firi<<" time in gcopter (in seconds): "<<elapsed_gcopter<<" time in ciri (in seconds): "<<elapsed_ciri<<std::endl;
        std::cout<<"hpolys size: "<<hpolys.size()<<std::endl;

        if (traj.getPieceNum() > 0)
        {
            
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Polytope Visualization"));
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
            std::vector<pcl::Vertices> polygons;

            for (const auto &hPoly : CIRI_hpolys) {
                Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
                geo_utils::enumerateVs(hPoly, vPoly);

                quickhull::QuickHull<double> tinyQH;
                const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
                const auto &idxBuffer = polyHull.getIndexBuffer();

                // Add vertices to PCL PointCloud with RGB color
                int baseIndex = cloud->size();
                for (int i = 0; i < vPoly.cols(); ++i) {
                    pcl::PointXYZRGB point;
                    point.x = vPoly(0, i);
                    point.y = vPoly(1, i);
                    point.z = vPoly(2, i);
                    // Assign a color to the point (e.g., red)
                    point.r = 255;
                    point.g = 0;
                    point.b = 0;
                    cloud->push_back(point);
                }

                // Add triangles to PCL PolygonMesh
                for (size_t i = 0; i < idxBuffer.size(); i += 3) {
                    pcl::Vertices triangle;
                    triangle.vertices.push_back(baseIndex + idxBuffer[i]);
                    triangle.vertices.push_back(baseIndex + idxBuffer[i + 1]);
                    triangle.vertices.push_back(baseIndex + idxBuffer[i + 2]);
                    polygons.push_back(triangle);
                }
            }

            // Convert to PolygonMesh
            pcl::PolygonMesh polyMesh;
            pcl::toPCLPointCloud2(*cloud, polyMesh.cloud);
            polyMesh.polygons = polygons;
            viewer->addPolygonMesh(polyMesh, "polytope_mesh_ciri_uncertain");
            std::cout<<"ciri hpolys size: "<<CIRI_hpolys.size()<<std::endl;
            // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>());
            // std::vector<pcl::Vertices> polygons2;

            // for (const auto &hPoly : CIRI_hpolys_deterministic) 
            // {
            //     Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
            //     geo_utils::enumerateVs(hPoly, vPoly);

            //     quickhull::QuickHull<double> tinyQH;
            //     const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
            //     const auto &idxBuffer = polyHull.getIndexBuffer();

            //     // Add vertices to PCL PointCloud with RGB color
            //     int baseIndex = cloud2->size();
            //     for (int i = 0; i < vPoly.cols(); ++i) {
            //         pcl::PointXYZRGB point;
            //         point.x = vPoly(0, i);
            //         point.y = vPoly(1, i);
            //         point.z = vPoly(2, i);
            //         // Assign a color to the point (e.g., red)
            //         point.r = 0;
            //         point.g = 255;
            //         point.b = 0;
            //         cloud2->push_back(point);
            //     }

            //     // Add triangles to PCL PolygonMesh
            //     for (size_t i = 0; i < idxBuffer.size(); i += 3) {
            //         pcl::Vertices triangle;
            //         triangle.vertices.push_back(baseIndex + idxBuffer[i]);
            //         triangle.vertices.push_back(baseIndex + idxBuffer[i + 1]);
            //         triangle.vertices.push_back(baseIndex + idxBuffer[i + 2]);
            //         polygons2.push_back(triangle);
            //     }
            // }

            // // Convert to PolygonMesh
            // pcl::PolygonMesh polyMesh2;
            // pcl::toPCLPointCloud2(*cloud2, polyMesh2.cloud);
            // polyMesh2.polygons = polygons2;
            // viewer->addPolygonMesh(polyMesh2, "polytope_mesh_ciri_deterministic");



            sfc_generator.visualizePolytopePCL(viewer, hpolys, filtered_cloud, path_rrt, nodelist, traj);
            // sfc_generator.visualizePolytopePCL(viewer, hpolys, input_cloud, path_rrt, nodelist, traj);
            int n = hpolys.size();
            
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr neighbour_pcd(new pcl::PointCloud<pcl::PointXYZRGBA>);

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tangent_obstacles(new pcl::PointCloud<pcl::PointXYZRGBA>);
            for(int i=0; i<int(t_obs.size()); i++)
            {
                auto tangentMat = t_obs[i];
                for(int j = 0; int(j<tangentMat.rows()); j++)
                {
                
                    pcl::PointXYZRGBA point_rgba;
                    if(tangentMat(j, 0) != INFINITY && tangentMat(j, 1) != INFINITY && tangentMat(j, 2) != INFINITY )
                    {
                        point_rgba.x = tangentMat(j,0);
                        point_rgba.y = tangentMat(j,1);
                        point_rgba.z = tangentMat(j,2);
                        
                        // Set the color fields, here we initialize them to a default value
                        point_rgba.r = 255; // Red channel
                        point_rgba.g = 0; // Green channel
                        point_rgba.b = 0; // Blue channel
                        point_rgba.a = 100; // Alpha channel
                    }
                    // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
                    // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres
                    tangent_obstacles->points.push_back(point_rgba);
                }
            }


            
            // for(int i=0; i<path_rrt.rows(); i++)
            // {
            //     Eigen::Vector3d path_pt(path_rrt(i,0), path_rrt(i,1), path_rrt(i,2));
            //     int c = 0;
            //     for(int j = 0; j<n; j++)
            //     {
            //         if(geo_utils::checkInterior(hpolys[j], path_pt))
            //         {
            //             c+=1;
            //         }
            //     }
            //     std::cout<<" number of polygons in which "<<path_pt.transpose()<<" lies = "<<c<<std::endl;
            // }  
            std::vector<Eigen::Vector3d> deepPoints;
            for(int j = 0; j < n-1; j++)
            {
                auto poly1 = hpolys[j];
                auto poly2 = hpolys[j+1];
                Eigen::Vector3d potential_pt;
                if(geo_utils::findDeepestPointOverlap(poly1, poly2, potential_pt))
                {
                    deepPoints.push_back(potential_pt);
                }
            }
            std::cout<<"[gcopter debug] number of deep points: "<<deepPoints.size()<<" hpolys size: "<<n<<std::endl;
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr path_skeleton_pcd(new pcl::PointCloud<pcl::PointXYZRGBA>);
            // std::cout<<"[mmd debug] path size for mmd: "<<path_rrt_mmd.rows()<<std::endl;
            for(int i=0; i<int(deepPoints.size()); i++)
            {
                pcl::PointXYZRGBA point_rgba;
                point_rgba.x = deepPoints[i].x();
                point_rgba.y = deepPoints[i].y();
                point_rgba.z = deepPoints[i].z();
                
                // Set the color fields, here we initialize them to a default value
                point_rgba.r = 255; // Red channel
                point_rgba.g = 0; // Green channel
                point_rgba.b = 0; // Blue channel
                point_rgba.a = 100; // Alpha channel
                // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
                // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres
                path_skeleton_pcd->points.push_back(point_rgba);
            }
            auto shortPath = gCopter.getShortPath();
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr gcopter_short_path(new pcl::PointCloud<pcl::PointXYZRGBA>);
            
            for(int i=0; i<int(shortPath.cols()); i++)
            {
                pcl::PointXYZRGBA point_rgba;
                point_rgba.x = shortPath(0, i);
                point_rgba.y = shortPath(1, i);
                point_rgba.z = shortPath(2, i);
                
                // Set the color fields, here we initialize them to a default value
                point_rgba.r = 255; // Red channel
                point_rgba.g = 165; // Green channel
                point_rgba.b = 0; // Blue channel
                point_rgba.a = 255; // Alpha channel
                // std::cout<<"sphere added to: "<<point_radius<<" coord: "<<Path_rrt(i,0)<<Path_rrt(i,1)<<Path_rrt(i,2)<<std::endl;
                // viewer->addSphere(point_rgba, point_radius, 0.0, 1.0, 0.0, std::to_string(i));  // Green spheres
                gcopter_short_path->points.push_back(point_rgba);
            }
            std::cout<<"[gcopter debug] number of rrt points: "<<path_rrt.rows()<<" number of gcopter short points: "<<shortPath.cols()<<std::endl;

            for(int i = 0; i<path_rrt.rows(); i++)
            {
                if(i >= deepPoints.size())
                {
                    std::cout<<i<<" th index rrt point not matching"<<std::endl;
                }
                else
                {
                    std::cout<<"rrt path point: "<<path_rrt(i,0)<<" "<<path_rrt(i,1)<<" "<<path_rrt(i,2)<<" deep point: "<<deepPoints[i].transpose()<<std::endl;
                }
            }
            // viewer->addPointCloud(path_skeleton_pcd, "geoutils deep points");
            // viewer->addPointCloud(gcopter_short_path, "gcopter init points");
            viewer->addPointCloud(tangent_obstacles, "tangent_points");
            // viewer->addPointCloud(neighbour_pcd,"neighbour_pcd");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "tangent_points");
            // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gcopter init points");

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(input_cloud, 255, 0, 0); // RGB: Red
            viewer->addPointCloud(input_cloud, "original_pcd");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "original_pcd");

            viewer->setBackgroundColor(0.0, 0.0, 0.0);
            viewer->addCoordinateSystem(1.0);
            viewer->setRepresentationToWireframeForAllActors();

            while (!viewer->wasStopped()) {
                viewer->spinOnce(100);
            }
            // sfc_generator.visualizePolytopePCL(hpolys, cloud_filtered, path_rrt, nodelist, new_traj);

        }
    }
    else
    {
        std::vector<NodePtr> nodeList = rrt_path_gen.getTree();
        std::cout<<"[no path debug] size of nodelist: "<<nodeList.size()<<std::endl;
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point cloud visualization"));
        int i = 0;
        for(NodePtr node : nodeList)
        {
            if(node->preNode_ptr != NULL)
            {
                pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]); // parent point 
                pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]); // child point 

                // Add a line between the two points
                viewer->addLine(point1, point2, 1.0, 0.0, 0.0, "line"+std::to_string(i++));
            }
        }
        
        viewer->addPointCloud(inflated_cloud, "original_pcd");
        // Optional: Set camera parameters and color
        viewer->setBackgroundColor(0.0, 0.0, 0.0);
        viewer->addCoordinateSystem(1.0);
        viewer->setRepresentationToWireframeForAllActors();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }

    }
    }

    
    return 0;
}