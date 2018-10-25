
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_others.hpp"


using namespace caffe;
int main(int argc, char** argv) {
  if (argc < 7) {
    LOG(ERROR)<< "get_curve_FDDB.bin"<<
				" gt_file  "<<
				" pred_file  "<<
				" overlap_threshold  "<<
				" outputfolder "<<
				" img_filder" <<
				" showing[0/1]";
    return 0;

  }

  int n_positive=0;
  std::cout<<"showing: "<<(strcmp("1",argv[6]) == 0 )<<std::endl;
  vector< std::pair<float, vector<float> > > pred_instances_with_gt;
  vector<float> precision, recall;
  float threshold = atof(argv[3]);
  GetPredictedWithGT_FDDB(argv[1], argv[2],pred_instances_with_gt,
                 n_positive,  strcmp("1",argv[6]) == 0, argv[5],argv[4],threshold);
  LOG(INFO)<<"AUC("<<threshold <<"): "<< GetTPFPPoint_FDDB(pred_instances_with_gt,n_positive, precision, recall)<< "  n_face = "<<
                   n_positive;
  std::ofstream pr_file((string(argv[4])+string("AUC.txt")).c_str());

        for (int i=0; i< precision.size(); i++) {
                pr_file << precision[i] << " " << recall[i]<< std::endl;
        //              LOG(INFO) << "p = " << p << ", r = " << r;
        }
        pr_file.close();

  return 0;
}
