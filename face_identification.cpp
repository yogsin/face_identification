void recognice_face(int argv, char **argc)
{
  //加载待识别的人脸特征
  std::string srcRoot = argc[2];
  std::string srcFe = srcRoot + "features/";
  std::string srcMat = srcRoot + "faces/";
  std::vector<std::string> all_fefile;
  std::vector<cv::Mat> all_mat;
  traversal_allfiles(srcFe, all_fefile);
  std::vector<float *> all_fe;
  std::vector<std::string> all_image_name;
  for (int i = 0; i < all_fefile.size(); ++i) {
    std::string fefile = srcFe + all_fefile[i];
    std::string head = getFileName(fefile, false);
    all_image_name.push_back(head);
    std::string matFile = srcMat + head + ".png";
    cv::Mat srcImage = cv::imread(matFile);
    all_mat.push_back(srcImage);
    std::vector<float> fe;
    loadFaceFea(fefile, fe);
    float *infe = new float[fe.size()];
    for (int k = 0; k < fe.size(); ++k) {
      infe[k] = fe[k];
    }
    all_fe.push_back(infe);
  }
  //加载特征完毕
  //加载人脸特征库
  std::vector<std::string> nameVec;
  traversal_single_dirs(registry_dir, nameVec);//遍历查找注册文件夹内的子文件夹（获取"人名/"）
  int RegNumber = nameVec.size();//获取注册人数
  printf("number of person is %d\n",RegNumber);
  std::vector<cv::Mat> all_lib_mat;
  printf("starting...\n");
  FILE *rec_result = fopen((srcRoot + "rec_info.txt").c_str(), "wb");
  //遍历所有待识别的人脸
  for (int n=0; n < all_fefile.size(); ++n){
    printf("%d\n",n);
    std::vector<float> identity_cores;//存放每个人比较的最高分
    //先遍历每个人，与每个人比较
    for (int i =0; i < RegNumber; ++i){
      std::vector<float *> all_lib_fe;
      printf("person %d\n",i);
      std::string srcLibFe;
      std::string srcLibMat;
      printf("dir %s\n", nameVec[i].c_str());
      srcLibFe = registry_dir + nameVec[i] + "features/";
      srcLibMat = registry_dir + nameVec[i] + "faces/";
      printf("dir %s\n", srcLibFe.c_str());
      std::vector<std::string> all_lib_fefile;
      traversal_allfiles(srcLibFe, all_lib_fefile);
      //遍历一个人的所有人脸库
      for (int j = 0; j < all_lib_fefile.size(); ++j) {
        std::string lib_fefile = srcLibFe + all_lib_fefile[j]; 
        printf("current txt %s\n",lib_fefile.c_str());
        std::vector<float> lib_fe;
        loadFaceFea(lib_fefile, lib_fe);
        float *lib_infe = new float[lib_fe.size()];
        for (int k = 0; k < lib_fe.size(); ++k) {
          lib_infe[k] = lib_fe[k];
        }
        all_lib_fe.push_back(lib_infe);
      }
      printf("len %d\n",all_lib_fe.size());
      //计算相似度
      float *tempFe = new float[FEATURELENGTH];
      int toSee = n;
      memcpy(tempFe, all_fe[n], sizeof(float) * FEATURELENGTH);//select num n;
      std::vector<float> scores;
      faceScores(all_lib_fe, tempFe, scores);
      //找最大值
      std::vector<float>::iterator biggest = std::max_element(std::begin(scores), std::end(scores));
      identity_cores.push_back(*biggest);
      printf("Max element is %f\n", *biggest);

	  std::vector<float> scores_test;
	  scores_test.push_back(2.0);
	  scores_test.push_back(3.0);
	  scores_test.push_back(1.0);
	  printf("scores_test %f\n",scores_test[0]);
      //找最大值
      std::vector<float>::iterator biggest_test = std::max_element(std::begin(scores_test), std::end(scores_test));
      printf("Max element is %f\n", *biggest_test);

      //std::cout << "Max element is " << *biggest<< " at position " << std::distance(std::begin(scores), biggest) << std::endl;
      //all_lib_fe.clear();
    }
    std::vector<float>::iterator biggest = std::max_element(std::begin(identity_cores), std::end(identity_cores));
    //std::cout << "Max element is " << *biggest<< " at position " << std::distance(std::begin(identity_cores), biggest) << std::endl;
    printf("it is %s\n", nameVec[std::distance(std::begin(identity_cores), biggest)].c_str());
    fprintf(rec_result, "%s\t%s\n", (srcFe + all_fefile[n]).c_str(),nameVec[std::distance(std::begin(identity_cores), biggest)].substr(0, nameVec[std::distance(std::begin(identity_cores), biggest)].length()-1).c_str());
  }
  fclose(rec_result);
  //加载人脸特征库完毕
  printf("done\n");
}
