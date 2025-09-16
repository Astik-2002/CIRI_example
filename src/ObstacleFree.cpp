#include "ObstacleFree.hpp"
#include "corridor_finder_dynamic.h"
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

std::vector<Eigen::Matrix3d> ObstacleFree::loadCSVToBbox(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<Eigen::Matrix3d> dynamic_bbox;
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return dynamic_bbox;
    }

    std::string line;
    // Skip the header line if there's any
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string c_x, c_y, c_z;
        std::string v_x, v_y, v_z;
        std::string h, l, w;

        std::getline(ss, c_x, ',');
        std::getline(ss, c_y, ',');
        std::getline(ss, c_z, ',');
        std::getline(ss, h, ',');
        std::getline(ss, l, ',');
        std::getline(ss, w, ',');
        std::getline(ss, v_x, ',');
        std::getline(ss, v_y, ',');
        std::getline(ss, v_z, ',');
        
        Eigen::Matrix3d mat;
        mat << std::stod(c_x), std::stod(c_y), std::stod(c_z),
               std::stod(v_x),   std::stod(v_y),   std::stod(v_z),
               std::stod(h), std::stod(l), std::stod(w);

        dynamic_bbox.push_back(mat);
    }

    return dynamic_bbox;
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

        if (ciri.convexDecomposition(bd, pc, a, b) != super_utils::SUCCESS) 
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


void ObstacleFree::visualizePolytopePCL(pcl::visualization::PCLVisualizer::Ptr &viewer, const std::vector<Eigen::MatrixX4d> &hPolys, pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, Eigen::MatrixXd &Path_rrt, const std::vector<NodePtr> nodelist, const Trajectory<5> &traj) 
{

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

void ObstacleFree::visualizePolytopePCL(pcl::visualization::PCLVisualizer::Ptr &viewer, const std::vector<Eigen::MatrixX4d> &hPolys, pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud, Eigen::MatrixXd &Path_rrt, const std::vector<NodePtr_dynamic> nodelist, const Trajectory<5> &traj) 
{

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

bool ObstacleFree::isNodeCollisionFree(NodePtr_dynamic node,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr& static_cloud,
                         const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& dynamic_points,
                         double t,
                         double safe_radius)
{
    const Eigen::Vector3d pos = node->coord.head<3>();
    const double radius_sq = safe_radius * safe_radius;

    // 1. Check against static obstacles
    for (const auto& pt : static_cloud->points)
    {
        Eigen::Vector3d obst(pt.x, pt.y, pt.z);
        if ((obst - pos).squaredNorm() <= radius_sq)
            return false; // collision with static obstacle
    }

    // 2. Check against dynamic obstacles at time t
    for (const auto& dyn_pair : dynamic_points)
    {
        Eigen::Vector3d dyn_pos = dyn_pair.first + dyn_pair.second * t;
        if ((dyn_pos - pos).squaredNorm() <= radius_sq)
            return false; // collision with dynamic obstacle
    }

    return true; // no collision
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

void ObstacleFree::visualizeTemporalCIRI(
    const std::vector<Eigen::MatrixX4d> &hpolys, 
    const std::vector<double> &time_stamps,
    pcl::visualization::PCLVisualizer::Ptr &viewer,
    const std::string &base_name) 
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<pcl::Vertices> polygons;

    // Same palette + gradient as obstacles
    static const std::vector<std::tuple<int, int, int>> color_palette = {
                {255, 0, 0},  
                {213, 42, 42},    
                {171, 84, 84},      
                {129, 126, 126},    
                {87, 168, 168}       
            };

    auto applyGradient = [](int r, int g, int b, double factor) {
        int nr = std::clamp(int(r * factor), 0, 255);
        int ng = std::clamp(int(g * factor), 0, 255);
        int nb = std::clamp(int(b * factor), 0, 255);
        return std::make_tuple(nr, ng, nb);
    };

    const int NUM_COLOR_STEPS = 5;  // Number of distinct colors to cycle through
    const int COLORS_PER_STEP = std::max(3, int((time_stamps.size() + 1) / NUM_COLOR_STEPS));
    int ciri_idx = 0;

    for (size_t poly_idx = 0; poly_idx < hpolys.size(); ++poly_idx) {
        const auto &hPoly = hpolys[poly_idx];

        // Choose base color based on time bin
        int color_index = (ciri_idx / COLORS_PER_STEP) % color_palette.size();
        auto [r_base, g_base, b_base] = color_palette[color_index];

        // Gradient factor within bin
        int step_offset = ciri_idx % COLORS_PER_STEP;
        double gradient_factor = 0.5 + 0.5 * (double(step_offset) / std::max(1, COLORS_PER_STEP - 1));
        auto [r, g, b] = applyGradient(r_base, g_base, b_base, gradient_factor);

        // Extract vertices
        Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
        geo_utils::enumerateVs(hPoly, vPoly);

        // Convex hull
        quickhull::QuickHull<double> tinyQH;
        const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
        const auto &idxBuffer = polyHull.getIndexBuffer();

        // Add vertices with color
        int baseIndex = cloud->size();
        for (int i = 0; i < vPoly.cols(); ++i) {
            pcl::PointXYZRGB point;
            point.x = vPoly(0, i);
            point.y = vPoly(1, i);
            point.z = vPoly(2, i);
            point.r = r;
            point.g = g;
            point.b = b;
            cloud->push_back(point);
        }

        // Add triangles
        for (size_t i = 0; i < idxBuffer.size(); i += 3) {
            pcl::Vertices triangle;
            triangle.vertices.push_back(baseIndex + idxBuffer[i]);
            triangle.vertices.push_back(baseIndex + idxBuffer[i + 1]);
            triangle.vertices.push_back(baseIndex + idxBuffer[i + 2]);
            polygons.push_back(triangle);
        }

        ++ciri_idx;  // advance color counter per poly
    }

    // Build polygon mesh
    pcl::PolygonMesh polyMesh;
    pcl::toPCLPointCloud2(*cloud, polyMesh.cloud);
    polyMesh.polygons = polygons;
    
    viewer->addPolygonMesh(polyMesh, base_name);
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, base_name);
}

// Add this function to your ObstacleFree class
std::tuple<int, int, int> ObstacleFree::timeToColor(double time, double max_time, bool use_gradient) 
{
    // Define the specific color palette in order
    static const std::vector<std::tuple<int, int, int>> color_palette = {
        {255, 105, 180},  // Pink
        {255, 255, 0},    // Yellow
        {0, 0, 255},      // Blue
        {0, 255, 0},      // Green
        {255, 0, 0}       // Red
    };
    
    if (!use_gradient) {
        // Fixed color for time ranges
        double normalized_time = std::fmod(time, max_time) / max_time;
        int color_index = std::min(static_cast<int>(normalized_time * color_palette.size()), 
                                  static_cast<int>(color_palette.size() - 1));
        return color_palette[color_index];
    }
    
    // Gradient blending between colors
    double normalized_time = std::fmod(time, max_time) / max_time;
    double segment = normalized_time * (color_palette.size() - 1);
    int segment_index = static_cast<int>(segment);
    double blend_factor = segment - segment_index;
    
    if (segment_index >= color_palette.size() - 1) {
        return color_palette.back();
    }
    
    auto [r1, g1, b1] = color_palette[segment_index];
    auto [r2, g2, b2] = color_palette[segment_index + 1];
    
    int r = static_cast<int>(r1 * (1 - blend_factor) + r2 * blend_factor);
    int g = static_cast<int>(g1 * (1 - blend_factor) + g2 * blend_factor);
    int b = static_cast<int>(b1 * (1 - blend_factor) + b2 * blend_factor);
    
    return {r, g, b};
}

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

Eigen::Matrix3Xd ObstacleFree::getObstaclePoints_continous(double &t1, double &t2, double PCDstart_time, pcl::PointCloud<pcl::PointXYZ> cloud_input, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points, Eigen::Matrix<double, 6, 4> &bd)
{
    Eigen::Matrix3Xd obstacle_points(3,0);
    int k = cloud_input.points.size();
    for (int i=0; i<k; i++)
    {
        auto pt = cloud_input.points[i];
        Eigen::Vector3d pos(pt.x, pt.y, pt.z);
        if ((bd.leftCols<3>() * pos + bd.rightCols<1>()).maxCoeff() < 0.0)
        {
            obstacle_points.conservativeResize(3, obstacle_points.cols() + 1);
            obstacle_points.col(obstacle_points.cols() - 1) = pos;
        }
    }
    // std::cout<<"input cloud size: "<<k<<std::endl;
    for (const auto& [position, velocity] : dynamic_points)
    {
        for(double t = t1; t <= t2; t += 0.2)
        {
            Eigen::Vector3d pos = position + (t-PCDstart_time)*velocity;
            if ((bd.leftCols<3>() * pos + bd.rightCols<1>()).maxCoeff() < 0.0)
            {
                obstacle_points.conservativeResize(3, obstacle_points.cols() + 1);
                obstacle_points.col(obstacle_points.cols() - 1) = pos;
            }
        }
    }

    return obstacle_points;
}

void ObstacleFree::convexCoverCIRI_dynamic(
        pcl::PointCloud<pcl::PointXYZ> cloud_input, 
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points,
        const std::vector<Eigen::Vector4d> &path, 
        const Eigen::Vector3d &lowCorner,
        const Eigen::Vector3d &highCorner,
        const double &range,
        std::vector<Eigen::MatrixX4d> &hpolys,
        double PCDStart_time,
        double _uav_radius,
        std::vector<double> &time_vector_poly,
        const double eps)
    {
        super_planner::CIRI ciri;

        hpolys.clear();
        int n = int(path.size());
        
        Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
        bd(0, 0) = 1.0;
        bd(1, 0) = -1.0;
        bd(2, 1) = 1.0;
        bd(3, 1) = -1.0;
        bd(4, 2) = 1.0;
        bd(5, 2) = -1.0;

        Eigen::MatrixX4d hp, gap;
        Eigen::MatrixX3d tangent_pcd1, tangent_pcd2;
        Eigen::Vector3d a(path[0][0], path[0][1], path[0][2]);
        Eigen::Vector3d b = a;
        time_vector_poly.clear();
        std::vector<Eigen::Vector3d> valid_pc;
        std::vector<Eigen::Vector3d> bs;
        valid_pc.reserve(cloud_input.points.size());
        ciri.setupParams(_uav_radius, 4); // Setup CIRI with robot radius and iteration number
        for (int i = 1; i < n;)
        {
            Eigen::Vector3d path_point = path[i].head<3>();
            a = b;
            b = path_point;
            double t1 = 0.0;
            if(i != 0)
            {
                t1 = path[i-1][3];
            }
            double t2 = path[i][3];
            time_vector_poly.push_back(t2-t1);
            i++;
            bs.emplace_back(b);

            bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
            bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
            bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
            bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
            bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
            bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

            valid_pc.clear();
            t1 -= PCDStart_time;
            t2 -= PCDStart_time;
            Eigen::Matrix3Xd pc = getObstaclePoints_continous(t1, t2, PCDStart_time, cloud_input, dynamic_points, bd);

            if (pc.cols() == 0) {
                Eigen::MatrixX4d temp_bp = bd;
                hpolys.emplace_back(temp_bp);
                continue;
            }
            if (ciri.convexDecomposition(bd, pc, a, b) != super_utils::SUCCESS) {
                std::cerr << "CIRI decomposition failed." << std::endl;
                time_vector_poly.pop_back();
                continue;
            }

            geometry_utils::Polytope optimized_poly;
            ciri.getPolytope(optimized_poly);
            hp = optimized_poly.GetPlanes();

            if (hpolys.size() != 0)
            {
                const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
                if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() +
                            ((hpolys.back() * ah).array() > -eps).cast<int>().sum())
                {   
                    if (ciri.convexDecomposition(bd, pc, a, a) != super_utils::SUCCESS) 
                    {
                        std::cerr << "CIRI decomposition failed." << std::endl;
                        continue;
                    }
                    time_vector_poly.push_back(2.0);
                    ciri.getPolytope(optimized_poly);
                    gap = optimized_poly.GetPlanes();
                    hpolys.emplace_back(gap);
                }
            }
            hpolys.emplace_back(hp);
        }
    }

void ObstacleFree::pcd_segmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<Eigen::Matrix3d> bboxes, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &dynamic_points)
{
    dynamic_points.clear();
    std::cout<<"seg check1"<<std::endl;
    // Efficient segmentation and filtering using std::remove_if

    auto& points = cloud_in->points;
    std::cout<<"seg check2"<<std::endl;

    auto new_end = std::remove_if(points.begin(), points.end(),
        [&](const pcl::PointXYZ& point) {
            for (const auto& bbox : bboxes)
            {
                double cx = bbox(0, 0), cy = bbox(0, 1), cz = bbox(0, 2);
                double half_height = bbox(2, 0) / 2.0;
                double half_length = bbox(2, 1) / 2.0;
                double half_width  = bbox(2, 2) / 2.0;
                Eigen::Vector3d bbox_velocity(bbox(1, 0), bbox(1, 1), bbox(1, 2));
                if (point.x >= cx - half_length && point.x <= cx + half_length &&
                    point.y >= cy - half_width  && point.y <= cy + half_width &&
                    point.z >= cz - half_height && point.z <= cz + half_height)
                {
                    dynamic_points.emplace_back(Eigen::Vector3d(point.x, point.y, point.z), bbox_velocity);
                    return true;  // Remove from static cloud
                }
            }
            return false;  // Keep in static cloud
        });
    std::cout<<"seg check3"<<std::endl;

    points.erase(new_end, points.end());  
    std::cout<<"seg check4"<<std::endl;
  
    points.shrink_to_fit();
    std::cout<<"seg check5"<<std::endl;

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
    Trajectory<5> traj_fixedtime;
    safeRegionRrtStarDynamic rrt_dynamic;
    safeRegionRrtStar rrt_path_gen;
    // rrt gen parameters
    float _uav_size = 0.5;
    float _safety_margin = 0.7;
    float _search_margin = 0.2;
    float _max_radius = 5.0;
    float _sensing_range = 25.0;

    float x_l = 0.0;
    float x_h = 45.0;
    float y_l = -15.0;
    float y_h = 15.0;
    float z_l = 0.5;
    float z_l2 = 0.5;
    float z_h = 5.5;
    float z_h2 = 5.5;
    float local_range = 20.0, sample_portion=0.25, goal_portion=0.1;
    int max_iter=100000;
    std::string folder_name = argv[1];
    std::string csv_file = folder_name + "/pcd_" + folder_name.substr(folder_name.size() - 4) + ".csv";
    std::string bbox_file = folder_name + "/bbox_array_" + folder_name.substr(folder_name.size() - 4) + ".csv";
    float voxelWidth = 0.25;
    float dilateRadius = 0.50;
    float leafsize = 0.50;
    const Eigen::Vector3i xyz((x_h - x_l) / voxelWidth,
                                  (y_h - y_l) / voxelWidth,
                                  (z_h2 - z_l2) / voxelWidth);

    const Eigen::Vector3d offset(x_l, y_l, z_l2);

    voxel_map::VoxelMap V_map(xyz, offset, voxelWidth);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> dynamic_points;

    auto input_cloud = sfc_generator.loadCSVToPointCloud(csv_file);
    auto bbox_dynamic = sfc_generator.loadCSVToBbox(bbox_file);

    std::cout<<"printing for debugging2"<<std::endl;
    if (!input_cloud) {
        std::cerr << "Failed to load point cloud from CSV." << std::endl;
        return -1;
    }
    
    if (bbox_dynamic.size() == 0) {
        std::cerr << "Failed to load bbox from CSV." << std::endl;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(leafsize, leafsize, leafsize); // Set the voxel grid size (adjust as needed)
    voxel_filter.filter(*filtered_cloud);

    
    std::cout<<"input pointcloud size before segmentation: "<<input_cloud->points.size()<<std::endl;

    sfc_generator.pcd_segmentation(input_cloud, bbox_dynamic, dynamic_points);

    std::cout<<"input pointcloud size after segmentation: "<<input_cloud->points.size()<<std::endl;
    std::cout<<"dynamic pointcloud size after segmentation: "<<dynamic_points.size()<<std::endl;

    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d a(0.0, -2.0, 0.8);
    Eigen::Vector3d b(13.0, -2.0, 1.0);

    sfc_generator.pclToVoxelMap(input_cloud, V_map, dilateRadius );

    auto time_bef_voxel_gen = std::chrono::steady_clock::now();
    auto time_aft_voxel_gen = std::chrono::steady_clock::now();
    auto elapsed_voxel = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_voxel_gen - time_bef_voxel_gen).count()*0.001;
    std::cout<<"[voxel comparision] time taken in voxel dilation: "<<elapsed_voxel<<std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr V_map_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr inflated_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    auto time_bef_voxel_inf = std::chrono::steady_clock::now();

    sfc_generator.pointCloudInflation(input_cloud, inflated_cloud);
    
    auto time_aft_voxel_inf = std::chrono::steady_clock::now();
    auto elapsed_inf = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_voxel_inf - time_bef_voxel_inf).count()*0.001;
    std::cout<<"[voxel comparision] time taken in new inflation: "<<elapsed_inf<<std::endl;
    std::cout<<"input pointcloud size: "<<input_cloud->points.size()<<std::endl;
    std::cout<<"dilated pcd size: "<<inflated_cloud->points.size()<<std::endl;
    
    std::cout<<"input pointcloud size: "<<input_cloud->points.size()<<std::endl;
    std::cout<<"dilated pcd size: "<<inflated_cloud->points.size()<<std::endl;
    std::cout<<"filtered pcd: "<<filtered_cloud->points.size()<<std::endl;

    // Show the point cloud
    double weight_t = 0.01;
    double max_vel_rrt = 0.5;
    auto t_rrt_start = std::chrono::steady_clock::now();
    /*
    // rrt_path_gen.setInput(*input_cloud, origin);
    // rrt_path_gen.setParam(_safety_margin, _search_margin, _max_radius, _sensing_range);
    // rrt_path_gen.setStartPt(a, b);
    // rrt_path_gen.setPt(a, b, x_l, x_h, y_l, y_h, z_l, z_h, _sensing_range, max_iter, sample_portion, goal_portion);
    // rrt_path_gen.SafeRegionExpansion(0.1);
    */
    bool testing_cont = true;
    double delta_t = 1.0;
    rrt_dynamic.reset();
    rrt_dynamic.setParam(_safety_margin, _search_margin, delta_t, 20.0, 90, 90, false);
    rrt_dynamic.setStartPt(a, b);
    Eigen::Vector3d min_pt = a.cwiseMin(b);
    Eigen::Vector3d max_pt = a.cwiseMax(b);

    // Expand by 1.0 in each direction (so box size = 2.0)
    double x_l_bkup = min_pt.x() - 1.0;
    double x_h_bkup = max_pt.x() + 1.0;
    double y_l_bkup = min_pt.y() - 1.0;
    double y_h_bkup = max_pt.y() + 1.0;

    rrt_dynamic.setInputDynamic(*input_cloud, dynamic_points, a, 0.0);
    rrt_dynamic.setPt(a, b, x_l_bkup, x_h_bkup, y_l_bkup, y_h_bkup, z_l, z_h,
                             _sensing_range, max_iter, sample_portion, goal_portion, 0.0, max_vel_rrt, 1.5*max_vel_rrt, weight_t);
    std::cout<<"parameters set2"<<std::endl;
    rrt_dynamic.SafeRegionExpansion(0.5, 0.0, true);
    auto t_rrt_end = std::chrono::steady_clock::now();
    auto elapsed_rrt = std::chrono::duration_cast<std::chrono::milliseconds>(t_rrt_end - t_rrt_start).count()*0.001;
    std::cout<<"[time comp] RRT Time taken: "<<elapsed_rrt<<std::endl;

    auto path_radius_pair = rrt_dynamic.getPath();
    bool path_exist = rrt_dynamic.getPathExistStatus();

    bool ciri_testing = false;
    if(!testing_cont)
    {
        if(path_exist)
        {
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point cloud visualization"));
            auto pathList = rrt_dynamic.getPathList();
            rrt_dynamic.resetRoot(pathList.size()-3);
            rrt_dynamic.SafeRegionRefine(0.2);
            rrt_dynamic.SafeRegionEvaluate(0.2);
            std::vector<NodePtr_dynamic> nodeList = rrt_dynamic.getTree();
            if(rrt_dynamic.getPathExistStatus())
            {
                std::vector<NodePtr_dynamic> pathList = rrt_dynamic.getPathList();
                // int color_step = std::max(1, int(255.0 / pathList.size())); // color spread
                const int NUM_COLOR_STEPS = 5;  // Number of distinct colors to cycle through
                const int COLORS_PER_STEP = std::max(3, int(pathList.size() / NUM_COLOR_STEPS));
                
                // Define a set of distinct colors that will cycle
                const std::vector<std::tuple<int, int, int>> color_palette = {
                    {87, 168, 168},  
                    {129, 126, 126},    
                    {171, 84, 84},      
                    {213, 42, 42},    
                    {255, 0, 0}       
                };
                int cloud_idx = 0;
                int line_idx = 0;
                std::cout<<"[path debug] number of nodes after expansion: "<<nodeList.size()<<std::endl;
                for (NodePtr_dynamic node : pathList)
                {
                    int color_index = (cloud_idx / COLORS_PER_STEP) % color_palette.size();
                    auto [r, g, b] = color_palette[color_index];
                    
                    // Normalize to 0-1 range
                    double r_norm = r / 255.0;
                    double g_norm = g / 255.0;
                    double b_norm = b / 255.0;
                    if (node->preNode_ptr != nullptr)
                    {
                        pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]);
                        pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]);
                        viewer->addLine(point1, point2, r_norm, g_norm, b_norm, "line_path" + std::to_string(line_idx++));
                    }
                    rrt_dynamic.checkingRadius(node);

                    // Add transparent sphere at node position with radius = node->radius
                    std::string sphere_id = "sphere_node_" + std::to_string(cloud_idx);
                    pcl::PointXYZ sphere_center(node->coord[0], node->coord[1], node->coord[2]);
                    // if(node->closest_static)
                    // {
                    //     viewer->addSphere(sphere_center, node->radius, 1.0, 0.5, 0.0, sphere_id); // orange color
                    // }
                    // else
                    // {
                    viewer->addSphere(sphere_center, node->radius, r_norm, g_norm, b_norm, sphere_id); // green color
                    // }
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, sphere_id);
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2.0, sphere_id);

                    // Compute dynamic obstacle point cloud at time t
                    double t = node->coord[3];
                    pcl::PointCloud<pcl::PointXYZ>::Ptr dyn_cloud_t(new pcl::PointCloud<pcl::PointXYZ>());
                    for (const auto& dyn_pair : dynamic_points)
                    {
                        Eigen::Vector3d predicted_pos = dyn_pair.first + dyn_pair.second * t;
                        dyn_cloud_t->points.emplace_back(predicted_pos[0], predicted_pos[1], predicted_pos[2]);
                    }

                    std::string cloud_id = "dyn_obs_t_" + std::to_string(cloud_idx);
                    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(
                    //     dyn_cloud_t,
                    //     (cloud_idx * color_step) % 256,
                    //     (128 + cloud_idx * color_step / 2) % 256,
                    //     (255 - cloud_idx * color_step) % 256
                    // );
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(
                        dyn_cloud_t,
                        r,
                        g,
                        b
                    );
                    viewer->addPointCloud(dyn_cloud_t, cloud_color, cloud_id);
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_id);

                    ++cloud_idx;
                }
                nodeList = rrt_dynamic.getTree();
                pathList = rrt_dynamic.getPathList();
                for (const auto& node : pathList)
                {
                    double t = node->coord[3];
                    bool collision = !sfc_generator.isNodeCollisionFree(node, input_cloud, dynamic_points, t, node->radius);
                    if (collision)
                    {
                        std::cout << "[Collision Detected] Node at time " << t << " with coord: " << node->coord.transpose() << std::endl;
                    }
                    else
                    {
                        std::cout << "[Collision sanity check] Node with coord: " << node->coord.transpose() <<" is collision free: "<< std::endl;

                    }
                }
                // int i = 0;
                // for(NodePtr_dynamic node : nodeList)
                // {
                //     if(node->preNode_ptr != NULL)
                //     {
                //         pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]); // parent point 
                //         pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]); // child point 

                //         // Add a line between the two points
                //         viewer->addLine(point1, point2, 1.0, 0.0, 0.0, "line"+std::to_string(i++));
                //     }
                // }
            }
            else
            {
                nodeList = rrt_dynamic.getTree();
                int i = 0;
                for(NodePtr_dynamic node : nodeList)
                {
                    if(node->preNode_ptr != NULL)
                    {
                        pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]); // parent point 
                        pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]); // child point 

                        // Add a line between the two points
                        viewer->addLine(point1, point2, 1.0, 0.0, 0.0, "line"+std::to_string(i++));
                    }
                }
            }

            // std::cout<<"[path debug] number of nodes after refine and evaluation: "<<nodeList.size()<<std::endl;
            std::cout<<"[path debug] path exist status: "<<rrt_dynamic.getPathExistStatus()<<std::endl;
            viewer->addPointCloud(input_cloud, "original_pcd");
            // Optional: Set camera parameters and color
            viewer->setBackgroundColor(1.0, 1.0, 1.0);
            viewer->addCoordinateSystem(1.0);
            viewer->setRepresentationToWireframeForAllActors();

            viewer->spin();
        }
        else
        {
            std::vector<NodePtr_dynamic> nodeList = rrt_dynamic.getTree();
            std::cout<<"[no path debug] size of nodelist: "<<nodeList.size()<<std::endl;
            std::cout<<"[path exist debug] number of dynamic points: "<<dynamic_points.size()<<std::endl;

            auto ptr = nodeList[0];
            std::cout<<"[no path debug] ptr params"<<ptr->coord.transpose()<<" : "<<ptr->radius<<std::endl;
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point cloud visualization"));
            int i = 0;
            for(NodePtr_dynamic node : nodeList)
            {
                if(node->preNode_ptr != NULL)
                {
                    pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]); // parent point 
                    pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]); // child point 

                    // Add a line between the two points
                    viewer->addLine(point1, point2, 1.0, 0.0, 0.0, "line"+std::to_string(i++));
                }
            }
            
            viewer->addPointCloud(input_cloud, "original_pcd");
            // Optional: Set camera parameters and color
            viewer->setBackgroundColor(0.0, 0.0, 0.0);
            viewer->addCoordinateSystem(1.0);
            viewer->setRepresentationToWireframeForAllActors();

            // while (!viewer->wasStopped()) {
            //     viewer->spinOnce(100);
            // }
            viewer->spin();
        }
    }
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

        b = a;

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

        viewer->spin();
    }
    else if(testing_cont)
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

            std::vector<Eigen::MatrixX4d> hpolys;
            std::vector<Eigen::MatrixX4d> hpolys_parallel;
            std::vector<Eigen::MatrixX3d> tangentObstacles;
            Eigen::Vector3d min_pt(x_l, y_l, z_l);
            Eigen::Vector3d max_pt(x_h, y_h, z_h);
            std::vector<NodePtr_dynamic> nodelist = rrt_dynamic.getTree();
            std::cout<<"nodelist size: "<<nodelist.size()<<std::endl;
            float progress = _max_radius;
            std::vector<Eigen::Vector3d> eigen_points;

            // V_map.getSurf(eigen_points);
            std::vector<Eigen::MatrixX4d> CIRI_hpolys;
            std::vector<Eigen::MatrixX4d> CIRI_hpolys_deterministic;
            std::vector<Eigen::Vector4d> corridor_points;
            std::vector<double> corridor_time_stamps;

            for(int i = 0; i<path_rrt.rows(); i++)
            {
                corridor_points.push_back(path_rrt.row(i));
                corridor_time_stamps.push_back(path_rrt(i,3));

            }
            
            std::vector<double> times;
            sfc_generator.convexCoverCIRI_dynamic(*input_cloud, dynamic_points, corridor_points, min_pt, max_pt, 1.0, hpolys, 0.0, _uav_size, times);
            Eigen::VectorXd time_vec_eig = Eigen::Map<Eigen::VectorXd>(times.data(), times.size());
            std::vector<Eigen::Vector3d> rrt_vec;
            for(int i = 0; i<path_rrt.rows(); i++)
            {
                Eigen::Vector3d pt(path_rrt(i, 0), path_rrt(i, 1), path_rrt(i, 2));
                rrt_vec.push_back(pt);
            }
            for(auto ele : times)
            {
                std::cout<<"times elements: "<<ele<<std::endl;
            }
            
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Polytope Visualization"));
            // sfc_generator.visualizeCIRI_gradient(hpolys, viewer);
            std::vector<NodePtr_dynamic> pathList = rrt_dynamic.getPathList();
            const int NUM_COLOR_STEPS = 5;  // Number of distinct colors to cycle through
            const int COLORS_PER_STEP = std::max(3, int(pathList.size() / NUM_COLOR_STEPS));
            sfc_generator.visualizeTemporalCIRI(hpolys, corridor_time_stamps, viewer, "id:0");
            // Define a set of distinct colors that will cycle

            const std::vector<std::tuple<int, int, int>> color_palette = {
                {255, 0, 0},  
                {213, 42, 42},    
                {171, 84, 84},      
                {129, 126, 126},    
                {87, 168, 168}       
            };
            auto applyGradient = [](int r, int g, int b, double factor) {
                int nr = std::clamp(int(r * factor), 0, 255);
                int ng = std::clamp(int(g * factor), 0, 255);
                int nb = std::clamp(int(b * factor), 0, 255);
                return std::make_tuple(nr, ng, nb);
            };

            int cloud_idx = 0;
            int line_idx = 0;

            std::cout << "[path debug] number of nodes after expansion: " << nodelist.size() << std::endl;

            for (NodePtr_dynamic node : pathList)
            {
                // Choose base color depending on step
                int color_index = (cloud_idx / COLORS_PER_STEP) % color_palette.size();
                auto [r_base, g_base, b_base] = color_palette[color_index];

                // Apply gradient within step
                int step_offset = cloud_idx % COLORS_PER_STEP;
                double gradient_factor = 0.5 + 0.5 * (double(step_offset) / std::max(1, COLORS_PER_STEP - 1));
                auto [r, g, b] = applyGradient(r_base, g_base, b_base, gradient_factor);

                // Normalize for addLine()
                double r_norm = r / 255.0;
                double g_norm = g / 255.0;
                double b_norm = b / 255.0;

                // Draw path edges
                if (node->preNode_ptr != nullptr)
                {
                    pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]);
                    pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]);
                    viewer->addLine(point1, point2, r_norm, g_norm, b_norm, "line_path" + std::to_string(line_idx++));
                }

                // Visualize dynamic obstacles at this node's time
                double t = node->coord[3];
                pcl::PointCloud<pcl::PointXYZ>::Ptr dyn_cloud_t(new pcl::PointCloud<pcl::PointXYZ>());
                for (const auto& dyn_pair : dynamic_points)
                {
                    Eigen::Vector3d predicted_pos = dyn_pair.first + dyn_pair.second * t;
                    dyn_cloud_t->points.emplace_back(predicted_pos[0], predicted_pos[1], predicted_pos[2]);
                }

                std::string cloud_id = "dyn_obs_t_" + std::to_string(cloud_idx);
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(dyn_cloud_t, r, g, b);
                viewer->addPointCloud(dyn_cloud_t, cloud_color, cloud_id);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_id);

                ++cloud_idx;
            }
            viewer->addPointCloud(input_cloud, "original_pcd");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "original_pcd");

            // viewer->setBackgroundColor(0.0, 0.0, 0.0);
            // viewer->addCoordinateSystem(1.0);
            // viewer->setRepresentationToWireframeForAllActors();
            // viewer->spin();

            gcopter::GCOPTER_PolytopeSFC gCopter;
            gcopter_fixed::GCOPTER_PolytopeSFC_FixedTime gcopter_fixedtime;
            Eigen::Vector3d front = a;
            int n = path_skeleton.rows();

            Eigen::Vector3d back = b;
            // GCopter parameters
            Eigen::Matrix3d iniState;
            Eigen::Matrix3d finState;
            iniState << front, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
            finState << back, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
            Eigen::VectorXd magnitudeBounds(5);
            Eigen::VectorXd penaltyWeights(5);
            Eigen::VectorXd physicalParams(6);
            std::vector<float> chiVec = {10000, 10000, 10000, 10000, 100000};
            magnitudeBounds(0) = 1.0;
            magnitudeBounds(1) = 2.1;
            magnitudeBounds(2) = 1.05;
            magnitudeBounds(3) = 2.0;
            magnitudeBounds(4) = 30.0;
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
            float relcostto1 = 1e-8;
            traj.clear();
            traj_fixedtime.clear();
            auto time_bef_gcopter = std::chrono::steady_clock::now();

            if (!gcopter_fixedtime.setup(iniState, finState, hpolys, INFINITY, smoothingEps, quadratureRes, magnitudeBounds, penaltyWeights, physicalParams, time_vec_eig))
            {
                std::cout<<"gcopter returned false during setup"<<std::endl;
            }
            if (std::isinf(gcopter_fixedtime.optimize(traj_fixedtime, relcostto1)))
            {
                std::cout<<"gcopter optimization cost is infinity"<<std::endl;
            }
            auto time_aft_gcopter = std::chrono::steady_clock::now();

            auto elapsed_gcopter = std::chrono::duration_cast<std::chrono::milliseconds>(time_aft_gcopter - time_bef_gcopter).count()*0.001;
            std::cout<<"hpolys size: "<<hpolys.size()<<std::endl;

            if (traj_fixedtime.getPieceNum() > 0 ) // && traj_fixedtime.getPieceNum() > 0)
            {
                std::cout<<"traj.getDurations: "<<traj.getDurations()<<std::endl;

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj_points(new pcl::PointCloud<pcl::PointXYZRGBA>());
                double T = 0.01; // Sampling interval
                Eigen::Vector3d lastX = traj_fixedtime.getPos(0.0);

                for (double t = T; t < traj_fixedtime.getTotalDuration(); t += T) {
                    Eigen::Vector3d X = traj_fixedtime.getPos(t);

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

                /*
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
                auto shortPath = gcopter_fixedtime.getShortPath();
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
                */
                
                // viewer->addPointCloud(path_skeleton_pcd, "geoutils deep points");
                // viewer->addPointCloud(gcopter_short_path, "gcopter init points");
                // viewer->addPointCloud(neighbour_pcd,"neighbour_pcd");
                // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gcopter init points");
                viewer->addPointCloud(traj_points, "trajectory_points_fixedtime");

                // viewer->addPointCloud(input_cloud, "original_pcd");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "trajectory_points_fixedtime");

                viewer->setBackgroundColor(1.0, 1.0, 1.0);
                viewer->addCoordinateSystem(1.0);
                // viewer->setRepresentationToWireframeForAllActors();

                viewer->spin();
            }
        }
        else
        {
            std::vector<NodePtr_dynamic> nodeList = rrt_dynamic.getTree();
            std::cout<<"[no path debug] size of nodelist: "<<nodeList.size()<<std::endl;
            auto ptr = nodeList[0];
            std::cout<<"[no path debug] ptr params"<<ptr->coord.transpose()<<" : "<<ptr->radius<<std::endl;
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point cloud visualization"));
            int i = 0;
            for(NodePtr_dynamic node : nodeList)
            {
                if(node->preNode_ptr != NULL)
                {
                    pcl::PointXYZ point1(node->preNode_ptr->coord[0], node->preNode_ptr->coord[1], node->preNode_ptr->coord[2]); // parent point 
                    pcl::PointXYZ point2(node->coord[0], node->coord[1], node->coord[2]); // child point 

                    // Add a line between the two points
                    viewer->addLine(point1, point2, 1.0, 0.0, 0.0, "line"+std::to_string(i++));
                }
            }
            
            viewer->addPointCloud(input_cloud, "original_pcd");
            // Optional: Set camera parameters and color
            viewer->setBackgroundColor(0.0, 0.0, 0.0);
            viewer->addCoordinateSystem(1.0);
            viewer->setRepresentationToWireframeForAllActors();

            // while (!viewer->wasStopped()) {
            //     viewer->spinOnce(100);
            // }
            viewer->spin();


        }

    }

    
    return 0;
}