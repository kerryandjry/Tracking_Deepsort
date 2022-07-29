#include <iostream>
#include <string>
#include <vector>

constexpr int N = 1 << 3;

struct Grid {
private:
  std::vector<int> v;
  size_t nx;

public:
  Grid(size_t nx, size_t ny) : v(nx * nx), nx(nx) {}

  int &operator()(size_t i, size_t j) { return v[j * nx + i]; }

  void print() {
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        std::cout << v[j * nx + i] << ' ';
      }
      std::cout << std::endl;
    }
  }
};

int main() {
  Grid g(N, N);
  size_t c = 0;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < N; j++) {
      g(i, j) = 0;
    }
  }
  for (int i = 1; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (j == 0)
        g(i, j) = 0;
      else
        g(i, j) = 1;
    }
  }

  g.print();

  return 0;
}
