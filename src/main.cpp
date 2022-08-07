#include <exception>
#include <iostream>
#include <memory>
#include <torch/torch.h>

#include "dataset.h"
#include "resnet.h"

void test_model() {

  torch::Device device("cpu");
  if (torch::cuda::is_available()) {
    std::cout << "using cuda" << std::endl;
    device = torch::Device("cuda:0");
  }

  torch::Tensor input = torch::randn({2, 3, 224, 224}).to(device);
  std::cout << "build net" << std::endl;
  ResNet<BasicBlock> resnet = resnet18();
  // std::cout << "net to device" << std::endl;
  resnet.to(device);

  torch::optim::Adam opt(resnet.parameters(), torch::optim::AdamOptions(0.001));

  torch::Tensor target = torch::randn({2, 1000}).to(device);

  for (size_t i = 0; i < 40; i++) {
    std::cout << "forward begin" << std::endl << std::flush;
    torch::Tensor output = resnet.forward(input);
    std::cout << "forward end " << output.sizes() << std::endl << std::flush;
    auto loss = torch::mse_loss(output.view({2, 1000}), target);
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    opt.step();
  }

  // std::cout << "forward net" << std::endl;
  // input = resnet.forward(input);
  // std::cout << input.sizes() << std::endl;
}

void dataset() {
  CIFAR10_Dataset cifar("/home/lab-509/Downloads/cifar-10-batches-bin/", false);

  std::shared_ptr<torch::Tensor> data;
  std::shared_ptr<torch::Tensor> label;

  cifar.GetBatchInTensor(2, data, label);

  std::cout << *data << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << *label << std::endl;
}

void forward_cpu() {
  torch::Device device("cpu");

  CIFAR10_Dataset cifar("/home/lab-509/Downloads/cifar-10-batches-bin/", false);

  std::shared_ptr<torch::Tensor> input;
  std::shared_ptr<torch::Tensor> target;

  std::cout << "build net" << std::endl;
  ResNet<BasicBlock> resnet = resnet18_cifar10();
  std::cout << "net to device" << std::endl;

  torch::optim::Adam opt(resnet.parameters(), torch::optim::AdamOptions(0.001));

  for (int i = 0; i < 5; i++) {
    cifar.GetBatchInTensor(2, input, target);

    std::cout << "forward begin" << std::endl << std::flush;
    torch::Tensor output = resnet.forward(*input);
    std::cout << "forward end " << output.sizes() << std::endl << std::flush;
    auto loss = torch::mse_loss(output.view({2, 10}), *target);
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    opt.step();
  }
}

void forward_gpu() {
  torch::Device device{"cuda:0"};

  CIFAR10_Dataset cifar{"/home/lab-509/Downloads/cifar-10-batches-bin/", false};

  std::shared_ptr<torch::Tensor> input;
  std::shared_ptr<torch::Tensor> target;

  std::cout << "build net" << std::endl;
   //ResNet<BasicBlock> resnet = resnet18_cifar10();
  auto resnet = std::make_shared<ResNet<BasicBlock>>(resnet18_cifar10());
  torch::optim::Adam opt{resnet->parameters(),
                         torch::optim::AdamOptions(0.001)};

  try {
    //torch::load(opt, "./net_adam.pt");
    torch::load(resnet, "./net.pt");
    std::cout << "load weights success!" << std::endl;
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }

  resnet->to(device);
  std::cout << "net to device" << std::endl;

  for (int i = 0; i < 500; i++) {
    cifar.GetBatchInTensor(16, input, target);
    // std::cout<<"forward begin"<<std::endl<<std::flush;
    torch::Tensor output = resnet->forward(input->to(device));
    // std::cout<<"forward end "<<output.sizes()<<std::endl<<std::flush;
    auto loss = torch::mse_loss(output.view({16 , 10}), target->to(device));
    std::cout << "Loss " << i << " : " << loss.item<float>() << std::endl;
    loss.backward();
    opt.step();
  }

  try {
    torch::save(resnet, "net.pt");
    torch::save(opt, "net_adam.pt");
    std::cout << "save model success!" << std::endl;
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
}

int main() {
  // test_model();
  //  dataset();
  forward_gpu();
}
