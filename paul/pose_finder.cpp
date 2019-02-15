#include <iostream>
#include <libconfig.h++>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

using namespace libconfig;

template<typename T>
std::vector<T> getCfgList(const Setting& setting) {
	size_t len = setting.getLength();
	std::vector<T> res(len);
	for(size_t i = 0; i < len; i++) {
		res[i] = setting[i];
	}
	return res;
}

const cv::Mat_<float> world_pts({8, 3}, {
        3.313, 4.824, 0,
        1.337, 5.325, 0,
        0, 0, 0,
        1.936, -0.501, 0,
        13.290, 5.325, 0,
        11.314, 4.824, 0,
        12.691, -0.501, 0,
        14.627, 0, 0
});

const cv::Mat_<float> cube({8, 3}, {
		0, 0, 0,
		0, 3, 0,
		3, 3, 0,
		3, 0, 0,
		0, 0, -3,
		0, 3, -3,
		3, 3, -3,
		3, 0, -3
});

static void draw_cube(cv::Mat &img, cv::Mat pts) {
	cv::Mat ipts;
	pts.convertTo(ipts, CV_32SC1);
	//std::cout << ipts(cv::Range(0, 4), cv::Range::all()) << std::endl;
	cv::drawContours(img, std::vector<cv::Mat>({ipts(cv::Range(0, 4), cv::Range::all())}), -1, cv::Vec3b(0, 255, 0), -3);
	for(int i = 0; i < 4; i++) {
		int j = i + 4;
		cv::line(img, ipts.at<cv::Point_<int> >(i, 0), ipts.at<cv::Point_<int> >(j, 0), cv::Vec3b(255, 0, 0), 3);
	}
	cv::drawContours(img, std::vector<cv::Mat>({ipts(cv::Range(4, 8), cv::Range::all())}), -1, cv::Vec3b(0, 0, 255), 3);
}


template <typename> struct Debug;

struct contour_info {
	size_t idx;
	double area;
};

struct side {
	int size;
	cv::Vec4f line;
};

struct corner {
	cv::Point2f pt;
	bool legit;
};

struct rect {
	std::vector<struct corner> corners;
	cv::Point2f center;
	cv::Vec2f up_vec;
	float area;
};


static std::vector<struct corner> rect_sort(float ltr, const struct rect &r){
	std::vector<struct corner> corners = r.corners;
	auto key = [&r, ltr](cv::Point2f point) -> double {
		cv::Vec2f v = point - r.center;
		return std::fmod((std::atan2(-v[1], v[0]) - ltr) + 2 * M_PI,  2 * M_PI);
	};
	std::sort(corners.begin(), corners.end(), 
			[&key](const struct corner &a, const struct corner &b) { return key(a.pt) < key(b.pt); }
			);
	return corners;
}

template<typename T>
static std::vector<T> reduce_vector(const std::vector<T> & in, int factor) {
	std::vector<T> reduced_loop(in.size() / factor);
	for(size_t i = 0; i < reduced_loop.size(); i++) {
		reduced_loop[i] = in[i*factor];
	}
	return reduced_loop;
}
//def coord_change(pt, v1, v2):
//    return np.linalg.solve(np.float32(
//        [[v1[0], v2[0]],
//         [v1[1], v2[1]]]), np.float32(pt).T).T
static cv::Point2f coordChange(cv::Point2f point, cv::Vec2f right) {
	cv::Mat_<float> out;
	cv::solve(cv::Mat_<float>({2, 2}, {
				right[0], -right[1],
				right[1], right[0]
				}), cv::Mat_<float>(point), out);
	return out.at<cv::Point2f>(0, 0);
}

int main(int argc, char *argv[]) {
	typedef std::chrono::high_resolution_clock perf_timer;
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " <config file> <capture index> [<exposure override>]" << std::endl;
		return 1;
	}
	Config cfg;
	cfg.setAutoConvert(true);
	try {
		cfg.readFile(argv[1]);
	} catch(FileIOException &fioex) {
		std::cerr << "I/O error while reading file." << std::endl;
		return 1;
	} catch(ParseException &pex) {
	    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
				<< " - " << pex.getError() << std::endl;
		return 1;
	}
	world_pts(cv::Range::all(), cv::Range(1, 2)) *= -1;

	const Setting& root = cfg.getRoot();
	std::vector<float> distorts = getCfgList<float>(root["distorts"]);
	std::vector<float> l_cam_mtrx = getCfgList<float>(root["cam_mtrx"]);
	cv::Mat cam_mtrx = cv::Mat(l_cam_mtrx).reshape(0, 3);
	std::cout << cam_mtrx << std::endl;

	const Setting& thresh_hue = root["hsv_threshold_hue"];
	const Setting& thresh_saturation = root["hsv_threshold_saturation"];
	const Setting& thresh_value = root["hsv_threshold_value"];

	std::vector<float> mins = {thresh_hue[0], thresh_saturation[0], thresh_value[0]};
	std::vector<float> maxs = {thresh_hue[1], thresh_saturation[1], thresh_value[1]};

	int exposure = root["exposure"];

	std::vector<int> intTargetColor = getCfgList<int>(root["targetColor"]);
	std::vector<uint8_t> targetColor (intTargetColor.begin(), intTargetColor.end());
	std::string targetColorSpace = root["targetColorSpace"];
	if (targetColorSpace != "Lab") {
		cv::Mat targetColorM = cv::Mat(targetColor).reshape(3, 1);
		cv::Mat targetColorLAB;
		if(targetColorSpace == "HSV") {
			cv::cvtColor(targetColorM, targetColorM, cv::COLOR_HSV2RGB);
			targetColorSpace = "RGB";
		}

		if(targetColorSpace == "RGB") cv::cvtColor(targetColorM, targetColorLAB, cv::COLOR_RGB2Lab);
		else {
			std::cerr << "Unknown color space " << targetColorSpace << std::endl;
			return 1;
		}
		cv::Mat tc = targetColorLAB.reshape(1);
		targetColor = {tc.at<uint8_t>(0), tc.at<uint8_t>(1), tc.at<uint8_t>(2)};
	}

	if (argc == 4) exposure = strtol(argv[3], nullptr, 10);
    //os.system('v4l2-ctl -d /dev/video{} -c exposure_auto=1 -c white_balance_temperature_auto=0 -c exposure_absolute={}'.format(DEV, exposure))
    //os.system('v4l2-ctl -d /dev/video{} -c focus_auto=0'.format(DEV))
    //os.system('v4l2-ctl -d /dev/video{} -c focus_absolute=0'.format(DEV))

	int cap_idx = strtol(argv[2], nullptr, 10);
	std::ostringstream cs1;
	cs1 << "v4l2-ctl -d /dev/video" << cap_idx << " -c exposure_auto=1 -c white_balance_temperature_auto=0 -c exposure_absolute=" << exposure;
	system(cs1.str().c_str());

	std::ostringstream cs2;
	cs2 << "v4l2-ctl -d /dev/video" << cap_idx << " -c focus_auto=0";
	system(cs2.str().c_str());

	std::ostringstream cs3;
	cs3 << "v4l2-ctl -d /dev/video" << cap_idx << " -c focus_absolute=0";
	system(cs3.str().c_str());

	cv::VideoCapture cap(cap_idx);
	int w = 640, h = 480;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
	//cap.set(cv::CAP_PROP_EXPOSURE, argc == 4 ? strtol(argv[3], nullptr, 10) : exposure);
	//cap.open(cap_idx);

	cv::Mat fr, fr_lab, hsv_frame, op, h_corners, greenscale_float, eroded_hsv_1, cm;
	cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
	while(true) {
		if(!cap.read(fr)) break;
		
		auto t1 = perf_timer::now();
		cv::cvtColor(fr, fr_lab, cv::COLOR_BGR2Lab);
	    //std::vector<int> shape = {img.cols, img.rows, img.channels()};
		std::vector<int> shape = {fr.size[0], fr.size[1], fr.channels()};
		//std::cout << cv::Mat_<size_t>(shape) << std::endl;
		//std::cout << shape[0] << ' ' << shape[1] << ' ' << shape[2] << std::endl;
		xt::xtensor<uint8_t, 3> xt_lab = xt::adapt(fr_lab.data, shape[0] * shape[1] * shape[2], xt::no_ownership(), shape);
		xt::xtensor<uint16_t, 2> greenscale_16 = xt::zeros<uint16_t>({shape[0], shape[1]});
		for(int i = 0; i < 3; i++) {
			//xt::xtensor<uint8_t, 2> fr = xt::abs(xt::view(xt_lab, xt::all(), xt::all(), i) - targetColor[i]);
			greenscale_16 += xt::abs(xt::view(xt_lab, xt::all(), xt::all(), i) - targetColor[i]);
			//Debug<decltype()>{};
		}
		xt::xtensor<float, 2> greenscale = 255 - xt::cast<float>(xt::clip(greenscale_16, 0, 255));
		std::vector<int> g_shape = {shape[0], shape[1]};
		cv::Mat greenscale_mat(g_shape, CV_32FC1, greenscale.data());

		//std::cout << "aaa" << std::endl;

		cv::cvtColor(fr, hsv_frame, cv::COLOR_BGR2HSV);
		cv::inRange(hsv_frame, mins, maxs, op);
		cv::cornerHarris(greenscale_mat, h_corners, 2, 3, 0.03);
		cv::Mat cm;
		cv::cvtColor(h_corners / 65536, cm, cv::COLOR_GRAY2BGR);

		cv::erode(op, eroded_hsv_1, kernel);
		cv::dilate(eroded_hsv_1, eroded_hsv_1, kernel);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(eroded_hsv_1, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		std::vector<struct contour_info> infos(contours.size());
		for(size_t i = 0; i < contours.size(); i++) {
			infos[i].idx = i;
			infos[i].area = cv::contourArea(contours[i]);
		}
		std::sort(infos.begin(), infos.end(), [](const auto& a, const auto& b) { return a.area < b.area; });
		double prevArea = 0;
		std::vector<struct rect> rectangles;
		for(auto &c : infos) {
			std::vector<cv::Point> loop = contours[c.idx];
			double area = c.area;

			if (loop.size() < 20) continue;
			if (area < prevArea / 2) break;
			prevArea = area;

			float lookahead_f = pow(area, .4) * .4;
			if (loop.size() > 100) {
				loop = reduce_vector(loop, 3);
				lookahead_f /= 3;
			} else if(loop.size() > 50) {
				loop = reduce_vector(loop, 2);
				lookahead_f /= 2;
			}/* else {
				loop = reduce_vector(loop, 2);
				lookahead_f /= 2;
			}*/

			int lookahead = (int)lookahead_f;

			std::vector<cv::Point_<float>> cpts(loop.size());
			//loop2.reserve(loop.size());
			for(int i = 0; i < loop.size(); i++) {
				cv::Point v1 = loop[i];
				cv::Point v2 = loop[(i + lookahead) % loop.size()];
				auto vec = cv::normalize(cv::Vec2f(cv::Vec2i(v2 - v1)));
				cpts[i] = cv::Point_<float>(vec);
				//loop2.push_back(vec);
			}
			cv::Mat labels;
			cv::kmeans(cpts, 4, labels, cv::TermCriteria(cv::TermCriteria::Type::COUNT + cv::TermCriteria::Type::EPS, 10, 1.0), 10, cv::KMEANS_RANDOM_CENTERS);
			int ll = labels.rows;
			cv::Mat label_stack;
		    cv::vconcat(labels, labels, label_stack);
			cv::vconcat(label_stack, labels, label_stack);
			cv::Mat km_kernel = cv::Mat::ones(lookahead / 2, 1, CV_8UC1);
			std::vector<cv::Vec3b> colors = {cv::Vec3b(0, 0, 255), cv::Vec3b(0, 255, 0), cv::Vec3b(255, 0, 0), cv::Vec3b(255, 255, 0)};
			std::vector<struct side> sides;
			sides.reserve(4);
			for(int i = 0; i < 4; i++) {
				cv::Mat n_idxs;
				cv::erode(label_stack == i, n_idxs, km_kernel);
				//cv::Mat i2 = n_idxs(cv::Range(ll - lookahead / 2, ll * 2 - lookahead / 2), cv::Range::all());
				cv::Mat i2 = n_idxs(cv::Range(ll - lookahead, ll * 2 - lookahead), cv::Range::all()); // ???
				int n_pts = (int)cv::sum(i2)[0];
				std::vector<cv::Point_<int>> pts;
				pts.reserve(n_pts);
				for(int j = 0; j < i2.rows; j++) {
					if (i2.at<int>(j, 0)) {
						pts.push_back(loop[j]);
						//cv::Point & v1 = loop[j];
						//cv::Point & v2 = loop[(j + lookahead) % loop.size()];
						//fr.at<cv::Vec3b>(v1.y, v1.x) = colors[i];
						//cv::line(fr, v1, v2, colors[i], 1, cv::LINE_AA);
					}
				}
				if (pts.size() < 4) continue;
				struct side side;
				cv::fitLine(pts, side.line, cv::DIST_L2, 0, 0.01, 0.01);
				side.size = n_pts;
				sides.push_back(side);
				//cv::line(fr, cv::Point2i(line[0] * -50 + line[2], line[1] * -50 + line[3]), cv::Point2i(line[0] * 50 + line[2], line[1] * 50 + line[3]), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
			}
			//std::cout << sides.size() << std::endl;
			if (sides.size() < 4) continue;
			std::sort(sides.begin(), sides.end(), [](const auto& a, const auto& b) { return a.size < b.size; });
			std::swap(sides[1], sides[2]);
			std::vector<struct corner> corners;
			std::vector<cv::Point2f> j_corners;
			std::vector<cv::Point2i> icorners;
			icorners.reserve(4);
			corners.reserve(4);
			j_corners.reserve(4);
			for(int i = 0; i < 4; i++) {
				const cv::Vec4f & s1 = sides[i].line;
				const cv::Vec4f & s2 = sides[(i+1)%4].line;
				cv::Mat_<float> matx({4, 4}, {
						s1[0], 0, -1, 0,
						s1[1], 0, 0, -1,
						0, s2[0], -1, 0,
						0, s2[1], 0, -1
				});
				cv::Mat_<float> vecb({4, 1}, {-s1[2], -s1[3], -s2[2], -s2[3]});
				cv::Mat res;
				cv::solve(matx, vecb, res);
				cv::Point_<float> pt(res.at<float>(2), res.at<float>(3));
				corners.push_back({.pt = pt, .legit = false});
				j_corners.push_back(pt);
			}
			float ca = cv::contourArea(j_corners);
			float short_side_length = cv::norm(corners[2].pt - corners[1].pt);
			//int box_size = std::min((int)(short_side_length / 2), 7);
			int box_size = 7;//std::min((int)(sqrt(ca) * .3), 7);
			for(int i = 0; i < 4; i++) {
				const cv::Point_<float> pt = corners[i].pt;
				cv::Point2i ipt(pt);
				//std::cout << pt << std::endl;

				if (pt.x >= -box_size / 2 && pt.x < fr.cols + box_size / 2 && pt.y >= -box_size / 2 && pt.y < fr.rows + box_size / 2) {
					int mx = std::max(ipt.x - box_size, 0);
					int my = std::max(ipt.y - box_size, 0);

					cv::Mat region = h_corners(cv::Range(my, std::min(ipt.y + box_size, h_corners.rows - 1)), cv::Range(mx, std::min(ipt.x + box_size, h_corners.cols - 1)));
					if (region.empty()) goto no_refine;
					double max;
					cv::minMaxIdx(region, nullptr, &max);
					cv::Mat msk = region > max * 0.1;

					cv::Mat labels, stats, centroids;
					int n_labels;
					if((n_labels = cv::connectedComponentsWithStats(msk, labels, stats, centroids)) <= 1) {
						cv::rectangle(cm, cv::Point(ipt.x - box_size, ipt.y - box_size), cv::Point(ipt.x + box_size, ipt.y + box_size), cv::Vec3f(0.0, 0.0, 1.0));
						goto no_refine;
					}
					unsigned c_sum = 0;
					unsigned c_idx = 0;
					for(size_t j = 1; j < n_labels; j++) {
						cv::Mat m2;
						region.copyTo(m2, labels == j);
						int s = cv::sum(m2)[0];
						if (s > c_sum){
							c_sum = s;
							c_idx = j;
						}
					}
					if (c_idx == 0) goto no_refine;
					double ux = centroids.at<double>(c_idx,0);
					double uy = centroids.at<double>(c_idx,1);
					cv::Mat_<float> csp({1, 2}, {(float)ux + mx, (float)uy + my});
					cv::rectangle(cm, cv::Point(ipt.x - box_size, ipt.y - box_size), cv::Point(ipt.x + box_size, ipt.y + box_size), cv::Vec3f(0.0, 1.0, 0.0));
					cv::cornerSubPix(greenscale_mat, csp, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::Type::COUNT + cv::TermCriteria::Type::EPS, 100, .001));
					//std::cout << ux << ' ' << uy << std::endl;
					cv::Point2f pp = csp.at<cv::Point2f>(0, 0);
					corners[i] = {.pt = pp, .legit = true};
					j_corners[i] = pp;
					icorners.push_back(cv::Point2i(pp));
				} else {
					cv::rectangle(cm, cv::Point(ipt.x - box_size, ipt.y - box_size), cv::Point(ipt.x + box_size, ipt.y + box_size), cv::Vec3f(1.0, 0.0, 0.0));
no_refine:
					;
					icorners.push_back(cv::Point2i(pt));
				}
			}
			cv::Point2f pta = (corners[3].pt + corners[0].pt) / 2;
			cv::Point2f ptb = (corners[1].pt + corners[2].pt) / 2;
			cv::Vec2f up_vec = ptb - pta;
			if (up_vec[1] < 0) up_vec *= -1;
			struct rect new_rec {
				.corners = corners,
				.center = (pta + ptb) / 2,
				.up_vec = up_vec,
				.area = (float)cv::contourArea(j_corners)
			};
			rectangles.push_back(new_rec);

			cv::drawContours(fr, std::vector<std::vector<cv::Point2i > >({icorners}), -1, cv::Vec3f(255, 255, 255), 1, cv::LINE_AA);



			/*for(int i = 0; i < loop.size(); i++) {
				cv::Point v1 = loop[i];
				cv::Point v2 = loop[(i + lookahead) % loop.size()];
				cv::line(fr, v1, v2, colors[labels.at<int>(i, 0)], 1, cv::LINE_AA);
			}*/			



			//std::cout << labels << std::endl;

		}

		std::vector<std::pair<std::vector<cv::Point2f>, std::vector<cv::Point3f> > > pairs;
		pairs.reserve(rectangles.size() / 2);
		if (rectangles.size() != 0)
		for(int i = 0; i < rectangles.size() - 1; i++) {
			const struct rect &r1_ = rectangles[i];
			std::sort(rectangles.begin() + i+1, rectangles.end(), [&r1_](const auto& a, const auto& b) { return cv::norm(a.center - r1_.center) < cv::norm(b.center - r1_.center); });
			for(int j = i+1; j < rectangles.size(); j++) {
				//std::cout << i << " " << j << std::endl;
				const struct rect &r2_ = rectangles[j];
				const struct rect &r1 = r1_.center.x < r2_.center.x ? r1_ : r2_;
				const struct rect &r2 = r1_.center.x < r2_.center.x ? r2_ : r1_;

				cv::Vec2f ltr_vec = cv::normalize(cv::Vec2f(r2.center - r1.center));
				//cv::Point2f f_vec = coordChange(ltr, r1.up_vec)
				float ltr = std::atan2(-ltr_vec[1], ltr_vec[0]);
				/*auto rect_sorter = [](const struct rect &r){
					std::vector<cv::Point2f> pts2 = r.corners;
					auto key = [r&](cv::Point2f point){
						cv::Vec2f v = point - r.center;
						return ((math.atan2(-v[1], v[0]) - ltr) + 2 * PI) % (2 * PI);
					};
					std::sort(pts2.begin(), pts.end(), 
							[](const cv::Point2f &a, const cv::Point2f &b) { return key(a) < key(b); }
							);
					return pts2;
				}*/
				std::vector<struct corner> pts = rect_sort(ltr, r1);
				std::vector<struct corner> pts2 = rect_sort(ltr, r2);
				pts.insert(pts.end(), pts2.begin(), pts2.end());
				if ((!pts[0].legit && !pts[1].legit && !pts[4].legit && !pts[5].legit) ||
					(!pts[2].legit && !pts[3].legit && !pts[6].legit && !pts[7].legit)) continue;
				int total_legit = 0;
				for(auto &v : pts) total_legit += v.legit;
				if (total_legit < 4) continue;

				std::vector<cv::Point3f> world_pts_s;
				std::vector<cv::Point2f> cam_pts_s;
				world_pts_s.reserve(8);
				cam_pts_s.reserve(8);
				for(int i = 0, j = 0; i < pts.size(); i++) {
					if (pts[i].legit) {
						cam_pts_s.push_back(pts[i].pt);
						world_pts_s.push_back(world_pts.at<cv::Vec3f>(i, 0));
					}
				}
				cv::Mat_<double> rvecs({3, 1}, {0, 0, 0});
				cv::Mat_<double> tvecs({3, 1}, {0, 0, 0});
				if(!cv::solvePnP(world_pts_s, cam_pts_s, cam_mtrx, distorts, rvecs, tvecs, false, cv::SOLVEPNP_EPNP)) continue;
				cv::Mat reprojected;
				cv::projectPoints(world_pts_s, rvecs, tvecs, cam_mtrx, distorts, reprojected);
				float err = 0;
				//std::cout << cam_pts_s.size() << std::endl;
				//std::cout << rvecs << std::endl << tvecs << std::endl;
				//std::cout << reprojected << std::endl;
				for(int k = 0; k < cam_pts_s.size(); k++) {
					//std::cout << cam_pts_s[k] << ' ' << reprojected.at<cv::Point2f>(k,0) << std::endl;
					err += cv::norm(cam_pts_s[k] - reprojected.at<cv::Point2f>(k,0));
				}
				err /= cam_pts_s.size() * sqrt(r1.area + r2.area);
				//std::cout << i << ' ' << j << ' ' << err << std::endl;
				if (err < .1) {
					for(int i = 0; i < pts.size(); i++) {
						cv::putText(fr, std::to_string(i), cv::Point(pts[i].pt), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Vec3b(255, 255, 255), 1, cv::LINE_AA);
					}
					cv::line(fr, cv::Point(r1.center), cv::Point(r2.center), cv::Vec3b(255, 255, 255), 1, cv::LINE_AA);
					std::swap(rectangles[j], *(rectangles.end() - 1));
					rectangles.pop_back();
					pairs.push_back(std::make_pair(cam_pts_s, world_pts_s));
					//std::cout << rvecs << std::endl;
					//std::cout << tvecs << std::endl;
					break;
				}
			}
		}

		for(const auto &p : pairs) {
			cv::Mat_<double> rvecs({3, 1}, {0, 0, 0}), tvecs({3, 1}, {0, 0, 0});
			std::vector<int> inl;
			cv::solvePnPRansac(p.second, p.first, cam_mtrx, distorts, rvecs, tvecs, false, 100, 8.0, 0.99, inl);
			if (inl.size() < 6) continue;
			//std::cout << rvecs << std::endl << tvecs << std::endl;
			cv::Mat proj;
			cv::projectPoints(cube, rvecs, tvecs, cam_mtrx, distorts, proj);
			draw_cube(fr, proj);
		}
		auto t2 = perf_timer::now();
		std::cout << "F: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

		cv::imshow("greenscale", greenscale_mat / 255);
		cv::imshow("corners", cm);
		cv::imshow("mask", op);
		cv::imshow("f1", fr);

		if(cv::waitKey(1) == 27) break;
	}

}
