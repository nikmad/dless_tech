#include <stdlib.h>
int main()
{
system("g++ main.cpp robot.cpp -o out");
system("./out");
system("python particleplot.py");
return 0;
}
