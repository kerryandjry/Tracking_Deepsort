#pragma once
#include <c10/util/ArrayRef.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <set>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <vector>

#include <c10/util/Optional.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

class Base_Dataset : public torch::data::Dataset<Base_Dataset> {
protected:
  std::string _filename;
  std::shared_ptr<std::vector<std::shared_ptr<torch::Tensor>>> _data;
  std::shared_ptr<std::vector<std::shared_ptr<torch::Tensor>>> _label;
  bool _part;
  int _num_all;

public:
  torch::optional<size_t> size() const { return _data->size(); }
  Base_Dataset(std::string filename, bool part = false, int num_all = 0)
      : _filename(filename), _part(part), _num_all(num_all) {
    _data = std::make_shared<std::vector<std::shared_ptr<torch::Tensor>>>();
    _label = std::make_shared<std::vector<std::shared_ptr<torch::Tensor>>>();
  }

  torch::data::Example<> get(size_t index) override {
    return {_data->at(index)->clone(), _label->at(index)->clone()};
  };
};

class CIFAR10_Dataset : public Base_Dataset {

  std::vector<torch::Tensor> _data_vec;
  std::vector<torch::Tensor> _label_vec;

  bool ReadCifar10(const std::string &dir, const std::string &batchName,
                   size_t imgCount) {
    const int PATCH_SIZE = 32;
    const int N_CHANEL = 3;
    const int LINE_LENGTH = PATCH_SIZE * PATCH_SIZE * N_CHANEL + 1;

    bool isSuccess = false;

    std::fstream fs(dir + batchName, std::ios::in | std::ios::binary);

    if (fs.is_open()) {
      std::cout << "open file success: " << batchName << std::endl;
      char buffer[LINE_LENGTH];
      for (size_t imgIdx = 0; imgIdx < imgCount; imgIdx++) {
        fs.read(buffer, LINE_LENGTH);
        int class_label = (int)buffer[0];
        cv::Mat red_img(32, 32, CV_8UC1, &buffer[1]);
        cv::Mat green_img(32, 32, CV_8UC1, &buffer[1025]);
        cv::Mat blue_img(32, 32, CV_8UC1, &buffer[2049]);
        std::vector<cv::Mat> bgrMats = {blue_img, green_img, red_img};
        cv::Mat rgb_img;
        cv::merge(bgrMats, rgb_img);

        cv::resize(rgb_img, rgb_img, cv::Size(224, 224));
        std::shared_ptr<torch::Tensor> tensor_image =
            std::make_shared<torch::Tensor>();
        std::shared_ptr<torch::Tensor> image_label =
            std::make_shared<torch::Tensor>();

        *image_label = torch::zeros({10});
        (*image_label)[class_label] = 1;

        *tensor_image = torch::from_blob(
            rgb_img.data, {rgb_img.rows, rgb_img.cols, 3}, torch::kByte);
        *tensor_image = tensor_image->permute({2, 0, 1});
        *tensor_image = tensor_image->toType(torch::kFloat);
        *tensor_image = tensor_image->div(255);
        *tensor_image = tensor_image->unsqueeze(0);

        _data->push_back(tensor_image);
        _label->push_back(image_label);
        _data_vec.push_back(*tensor_image);
        _label_vec.push_back(*image_label);
      }
      isSuccess = true;
    } else {
      std::cout << "open file fails: " << batchName << std::endl;
      isSuccess = false;
    }

    fs.close();
    return isSuccess;
  }

public:
  CIFAR10_Dataset(std::string file_dir, bool part = false, int num_all = 10000)
      : Base_Dataset(file_dir, part, num_all) {
    const std::string dir = file_dir;
    const std::string batch_names[6] = {"data_batch_1.bin", "data_batch_2.bin",
                                        "data_batch_3.bin", "data_batch_4.bin",
                                        "data_batch_5.bin", "test_batch.bin"};

    size_t ImgCountPerBatch = num_all;
    bool success = ReadCifar10(dir, batch_names[3], ImgCountPerBatch);
  }

  void GetBatchInTensor_Random(int batch_size,
                               std::shared_ptr<torch::Tensor> &data,
                               std::shared_ptr<torch::Tensor> &label) {
    std::set<int> ruler;
    std::uniform_int_distribution<unsigned> uni(0, _num_all);
    std::default_random_engine eng;

    while (ruler.size() < batch_size) {
      ruler.insert(uni(eng));
    }

    std::vector<torch::Tensor> tmp_data_vec;
    std::vector<torch::Tensor> tmp_label_vec;
    auto it = _data_vec.begin();
    auto itl = _label_vec.begin();
    for (; it != _data_vec.end(); it++, itl++) {
      tmp_data_vec.push_back(*it);
      tmp_label_vec.push_back(*itl);
    }

    data = std::make_shared<torch::Tensor>();
    label = std::make_shared<torch::Tensor>();

    *data = torch::cat((tmp_data_vec), 0);
    *label = torch::stack((tmp_label_vec), 0);
  }

  void GetBatchInTensor(int batch_size, std::shared_ptr<torch::Tensor> &data,
                        std::shared_ptr<torch::Tensor> &label) {
    std::uniform_int_distribution<unsigned> uni(0, _num_all - batch_size);
    std::default_random_engine eng;
    int rand_num = uni(eng);

    data = std::make_shared<torch::Tensor>();
    label = std::make_shared<torch::Tensor>();

    *data =
        torch::cat(torch::TensorList(_data_vec.data() + rand_num,
                                     _data_vec.data() + rand_num + batch_size),
                   0);
    *label = torch::stack(
        torch::TensorList(_label_vec.data() + rand_num,
                          _label_vec.data() + rand_num + batch_size),
        0);
  }
};
