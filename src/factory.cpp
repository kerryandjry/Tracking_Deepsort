#include <iostream>
#include <memory>

class animal
{
public:
    virtual void print() const {
        std::cout << "animal" << std::endl;
    }
};

class cat : public animal
{
public:
    void print() const override {
        std::cout << "cat" << std::endl;
    }
};

class dog : public animal
{
public:
    void print() const override {
        std::cout << "dog" << std::endl;
    }
};

class Factory
{
public:
    virtual std::unique_ptr<animal> createanimal() = 0;
};

class dog_factory : public Factory
{
public:
    std::unique_ptr<animal> createanimal() override {
        return std::make_unique<dog>();
    }
};

class test
{
public:
    test(std::unique_ptr<Factory> af): animal_fac(std::move(af)) {}
    void bottonclick() {
        auto new_animal = animal_fac->createanimal();
        new_animal->print();
    }
private:
    std::unique_ptr<Factory> animal_fac;
};

int main(int argc, char *argv[])
{
    auto t = test(std::make_unique<dog_factory>()); 
    t.bottonclick();
    return 0;
}
