// (C) 2013, Nikhil Madduri, <nikhil.madduri@stud-mail.uni-wuerzburg.de>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <fstream>
using std::ifstream;
#include <cstdlib> // for exit function
// This program reads values from the file 'example.dat'
// and echoes them to the display until a negative value
// is read.
int main()
{
   ifstream indata; // indata is like cin
   float num; // variable for input value
   float num_total = 0.0;
   int count = 0;
   indata.open("runningtime.txt"); // opens the file
   if(!indata) { // file couldn't be opened
      cerr << "Error: file could not be opened" << endl;
      exit(1);
   }
   
   indata >> num;
   while ( !indata.eof() ) { // keep reading until end-of-file
      cout << "The next number is " << num << endl;
      num_total += num;
      count++;
      indata >> num; // sets EOF flag if no value found
   }
   indata.close();
   cout << "End-of-file reached.." << endl;

   cout << "Average time/frame = " << num_total/count << "\n" << endl;
   cout << "Average fps = " << count*1000000/num_total << "\n" << endl;
   cout << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n" << endl;

   return 0;
}
