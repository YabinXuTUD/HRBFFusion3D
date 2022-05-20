//
// Created by robin on 22-6-18.
//

#include "HRBF_fusion.h"
#include <iostream>

using namespace std;

//This file to Test

int main(int argc, char * argv[])
{

    //first of all we need to reveal the pure point-based fusion
    //modify the elasticfusion file
    //1. repeat the basic point-cloud fusion function

    MainController mainController(argc, argv);

    mainController.launch();

    return 0;

}
