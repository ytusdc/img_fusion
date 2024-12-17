#include <iostream>
#include <fstream>
#include <string>

#include "include/common.hpp"

using namespace std;

int main(int argc, char* argv[])
{	
	string file_path = "./img_test/*";

	// stitch_v1(file_path);

	stitch_v2(file_path);

	return 0;
}