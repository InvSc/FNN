#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <string>
#include <stdarg.h>
#include <sys/stat.h>
#include <pthread.h>
//#include <direct.h>
#define random(x) rand()%(x)

using namespace std;

const int MAXN = 50;      // Max neurons in any layer
const int MAXPATS = 50000; // Max training patterns

// mlp paramaters
const long NumIts = 100;   // Max training iterations
//int NumHN;	   // Number of hidden layers
//int NumHN1;	   // Number of neurons in hidden layer 1
//int NumHN2;	   // Number of neurons in hidden layer 2
//int NumHN3;
//int NumHN4;	   // Number of neurons in hidden layer 4
float LrnRate = 0.6;
float Mtm1 = 1.2;       // Momentum(t-1)
float Mtm2 = 0.4;       // Momentum(t-2)
float ObjErr = 0.005;  // Objective error

// mlp weights
//float **w1, **w11, **w111; // 1st layer wts
//float **w2, **w22, **w222; // 2nd layer wts
//float **w3, **w33, **w333; // 3rd layer wts
//float **w4, **w44, **w444; // 4th layer wts
//float **w5, **w55, **w555; // 5th layer wts


int NumIPs = 4, NumOPs = 1, Ordering = 0; // NumTrnPats = 6000, NumTstPats = 6000,

//char Tmp[500];
//char FName[500];

void TrainNet2(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
               int NumIPs, int NumOPs, int NumPats, int Ordering, int NumHN1, int activation);

void TrainNet3(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
               float **w3, float **w33, float **w333, int NumIPs, int NumOPs, int NumPats, int Ordering, int NumHN1,
               int NumHN2, int activation);

void
TrainNet4(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
          float **w3, float **w33, float **w333, float **w4, float **w44, float **w444, int NumIPs, int NumOPs,
          int NumPats, int Ordering, int NumHN1, int NumHN2, int NumHN3, int activation);


void
TestNet2(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222, int NumIPs, int NumOPs,
         int NumPats, string &outPath, int NumHN1,
         int activation);

void TestNet3(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
              float **w3, float **w33, float **w333, int NumIPs, int NumOPs, int NumPats, string &outPath, int NumHN1, int NumHN2,
              int activation);

void TestNet4(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
         float **w3, float **w33, float **w333, float **w4, float **w44, float **w444, int NumIPs, int NumOPs,
         int NumPats, string &outPath, int NumHN1, int NumHN2, int NumHN3,
         int activation);

float **Aloc2DAry(int m, int n);

void Free2DAry(float **Ary2D, int n);

int CountLines(const char *filename);

void *TrainOneDataset(void *threadarg);

void
TrainOneState(string dataset, string prefix, string suffix, int NumHN, int NumHN1, int NumHN2, int NumHN3,
              int activation);

void InitIndex(int *IndexLists, int n);

void Randomize(int *list, int n);

void RandomSwap(int *list, int n);

double select_activate(int order, double x);

double gauss1(double x, double c, double sig);

double gauss2(double x, double a, double b, double r);

double sigmoid(double x);

void createDir(string path);

struct thread_data {
    int thread_id;
    string dataset;
    string prefix;
    string suffix;
    int activation;
};

int main() {
    int count = 4, activation = 1, NUM_THREADS = 4, rc, i;
    pthread_t threads[NUM_THREADS];
    struct thread_data td[NUM_THREADS];
    string datasets[4] = {"iofrol", "paintcontrol", "rails",
                          "gsdtsr"}, prefix = "/Users/sc/Desktop/FNN_cpp/", zero_count = "12", suffix = "_normal" + zero_count + ".csv";
//    char* buffer = getcwd(NULL, 0);
//    cout << buffer << endl;
    if (activation == 0) {
        createDir(prefix + "sigmoid/");
    } else if (activation == 1) {
        createDir(prefix + "guass1/");
    }

    for (i = 0; i < NUM_THREADS; i++) {
        cout << "main() : creating thread, " << i << endl;
        td[i].thread_id = i;
        td[i].dataset = datasets[i];
        td[i].prefix = prefix;
        td[i].suffix = suffix;
        td[i].activation = activation;
        rc = pthread_create(&threads[i], NULL,
                            TrainOneDataset, (void *) &td[i]);
        if (rc) {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }
    pthread_exit(NULL);
}

void *TrainOneDataset(void *threadarg) {
    struct thread_data *my_data = (struct thread_data *) threadarg;
    string dataset = my_data->dataset;
    string prefix = my_data->prefix;
    string suffix = my_data->suffix;
    int activation = my_data->activation;
    if (activation == 0) {
        createDir(prefix + "sigmoid/" + dataset + "/");
    } else if (activation == 1) {
        createDir(prefix + "guass1/" + dataset + "/");
    }
    int NumHN = 1;
    for (int i = 1; i <= 4; i++) {
        TrainOneState(dataset, prefix, suffix, NumHN, i * 10, 0, 0, activation);
    }
    NumHN++;
    for (int i = 1; i <= 4; i++) {
        for (int j = i; j <= 4; j++) {
            TrainOneState(dataset, prefix, suffix, NumHN, i * 10, j * 10, 0, activation);
        }
    }
    NumHN++;
    for (int i = 1; i <= 4; i++) {
        for (int j = i; j <= 4; j++) {
            for (int k = j; k <= 4; k++) {
                TrainOneState(dataset, prefix, suffix, NumHN, i * 10, j * 10, k * 10, activation);
            }
        }
    }
    pthread_exit(NULL);
}

void InitIndex(int *IndexLists, int n) {
    for (int i = 0; i < n; i++) {
        IndexLists[i] = i;
    }
}


void Randomize(int *list, int n) {  //Shuffle the N data randomly
    if (n > 1) {
        srand(time(NULL));
        for (int i = n - 1; i != 0; i--) {
            int j = random(i);
            int tmp = list[i];
            list[i] = list[j];
            list[j] = tmp;
        }
    }
}

void RandomSwap(int *list, int n) {  // Randomly swap 2 datasets
    if (n > 1) {
        srand(time(NULL));
        int idx1 = rand() % n;
        int idx2 = rand() % n;
        int t = list[idx1];
        list[idx1] = list[idx2];
        list[idx2] = t;
    }
}

// Trains 2 layer back propagation neural network
void TrainNet2(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
               int NumIPs, int NumOPs, int NumPats, int Ordering, int NumHN1, int activation) {
    // x[][]=>input data, d[][]=>desired output data
    float *h1 = new float[NumHN1];                         // O/Ps of hidden layer
    float *y = new float[NumOPs];                         // O/P of Net
    float *ad1 = new float[NumHN1];                         // HN1 back prop errors
    float *ad2 = new float[NumOPs];                         // O/P back prop errors
    float PatErr, MinErr, AveErr, MaxErr, PcntErr = 0.0; // Pattern errors
    int p, i, j;                                         // for loops indexes
    long ItCnt = 0;                                         // Iteration counter
    long NumErr = 0;                                     // Error counter (added for spiral problem)

    cout << "NetArch: IP:" << NumIPs << " H1:" << NumHN1 << " OP:" << NumOPs << endl;
    cout << "Params: " << "LrnRate: " << LrnRate << "  Mtm1: " << Mtm1 << "  Mtm2: " << Mtm2 << endl;
    cout << endl;

    // Allocate memory for weights
//    w1 = Aloc2DAry(NumIPs, NumHN1); // 1st layer wts
//    w11 = Aloc2DAry(NumIPs, NumHN1);
//    w111 = Aloc2DAry(NumIPs, NumHN1);
//    w2 = Aloc2DAry(NumHN1, NumOPs); // 2nd layer wts
//    w22 = Aloc2DAry(NumHN1, NumOPs);
//    w222 = Aloc2DAry(NumHN1, NumOPs);

    // Init wts between -0.5 and +0.5
    srand(time(0));
    cout << "NumIPs: " << NumIPs << "NumHN1: " << NumHN1 << endl;
    for (i = 0; i < NumIPs; i++)
        for (j = 0; j < NumHN1; j++) {
            w1[i][j] = w11[i][j] = w111[i][j] = float(rand()) / RAND_MAX - 0.5;
//            cout << "w1[i][j]: " << w1[i][j] << endl;
        }
    for (i = 0; i < NumHN1; i++)
        for (j = 0; j < NumOPs; j++)
            w2[i][j] = w22[i][j] = w222[i][j] = float(rand()) / RAND_MAX - 0.5;
    // Initailize the array
    int *IndexLists = new int[NumPats];
    InitIndex(IndexLists, NumPats);
    if (Ordering >= 2)
        Randomize(IndexLists, NumPats);

    cout << "Training mlp for " << NumPats << " iterations:" << endl;
    cout << setprecision(6) << setw(7) << " #" << setw(12) << "MinErr" << setw(12) << "AveErr" << setw(12) << "MaxErr"
         << setw(12) << "%Err" << endl;
    for (;;) {
        MinErr = 3.4e38;
        AveErr = 0;
        MaxErr = -3.4e38;
        NumErr = 0;

        if (Ordering == 0) {
        } else if (Ordering == 1) {
            Randomize(IndexLists, NumPats);
        } else if (Ordering == 2) {
            RandomSwap(IndexLists, NumPats);
        } else {

            if (NumErr <= 1 / 10 * NumPats)
                RandomSwap(IndexLists, NumPats);
        }
        //Training
        for (int idx = 0; idx < NumPats; idx++) {
            p = IndexLists[idx];

            for (i = 0; i < NumHN1; i++) {
                float in = 0;
                for (j = 0; j < NumIPs; j++) {
                    in += w1[j][i] * x[p][j];
//                    cout << "w1[j][i]: " << w1[j][i] << " x[p][j]: " << x[p][j] << endl;
                }
//                cout << "in: " << in << endl;
                h1[i] = select_activate(activation, in); // Sigmoid fn
//                cout << "h1: " << h1[i] << endl;
            }
            for (i = 0; i < NumOPs; i++) {
                float in = 0;
                for (j = 0; j < NumHN1; j++) {
                    in += w2[j][i] * h1[j];
                }
//                cout << "in: " << in << endl;
                y[i] = select_activate(activation, in); // Sigmoid fn
//                cout << "y[]: " << y[i] << endl;
            }
            // Cal error for this pattern
            PatErr = 0.0;
            for (i = 0; i < NumOPs; i++) {
                float err = y[i] - d[p][i]; // actual-desired O/P
                if (err > 0)
                    PatErr += err;
                else
                    PatErr -= err;
                NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) ||
                           (y[i] >= 0.5 && d[p][i] < 0.5)); //added for binary classification problem
            }
            if (PatErr < MinErr)
                MinErr = PatErr;
            if (PatErr > MaxErr)
                MaxErr = PatErr;
            AveErr += PatErr;

            // Learn pattern with back propagation
            for (i = 0; i < NumOPs; i++) { // Modify layer 2 wts
                ad2[i] = (d[p][i] - y[i]) * y[i] * (1.0 - y[i]);
                for (j = 0; j < NumHN1; j++) {
                    w2[j][i] += LrnRate * h1[j] * ad2[i] +
                                Mtm1 * (w2[j][i] - w22[j][i]) +
                                Mtm2 * (w22[j][i] - w222[j][i]);
                    w222[j][i] = w22[j][i];
                    w22[j][i] = w2[j][i];
                }
            }
            for (i = 0; i < NumHN1; i++) { // Modify layer 1 wts
                float err = 0.0;
                for (j = 0; j < NumOPs; j++)
                    err += ad2[j] * w2[i][j];
                ad1[i] = err * h1[i] * (1.0 - h1[i]);
                for (j = 0; j < NumIPs; j++) {
                    w1[j][i] += LrnRate * x[p][j] * ad1[i] +
                                Mtm1 * (w1[j][i] - w11[j][i]) +
                                Mtm2 * (w11[j][i] - w111[j][i]);
                    w111[j][i] = w11[j][i];
                    w11[j][i] = w1[j][i];
                }
            }
        } // end for each pattern
        ItCnt++;
        AveErr /= NumPats;
        PcntErr = NumErr / float(NumPats) * 100.0;

        cout.setf(ios::fixed | ios::showpoint);
        cout << setprecision(6) << setw(6) << ItCnt << ": " << setw(12) << MinErr << setw(12) << AveErr << setw(12)
             << MaxErr << setw(12) << PcntErr << endl;

        if ((AveErr <= ObjErr) || (ItCnt == NumIts))
            break;
    } // end main learning loop
    // Free memory
    delete[] IndexLists;
    delete[] h1;
    delete[] y;
    delete[] ad1;
    delete[] ad2;
}

// Trains 3 layer back propagation neural network
void TrainNet3(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
               float **w3, float **w33, float **w333, int NumIPs, int NumOPs, int NumPats, int Ordering, int NumHN1,
               int NumHN2, int activation) {

    float *h1 = new float[NumHN1];    // O/Ps of hidden layer 1
    float *h2 = new float[NumHN2];    // O/Ps of hidden layer 2
    float *y = new float[NumOPs];    // O/P of Net
    float *ad1 = new float[NumHN1]; // HN1 back prop errors
    float *ad2 = new float[NumHN2]; // HN2 back prop errors
    float *ad3 = new float[NumOPs]; // O/P back prop errors
    float PatErr;                    // Absolute error sum of the pattern
    float MinErr;                    // Minimum epoch error
    float AveErr;                    // Aveage error in one epoch
    float MaxErr;                    // maximum epoch error

    int p, i, j;     // for loops indexes
    long ItCnt = 0;     // Iteration counter
    long NumErr = 0; // Error counter (added for spiral problem)

    cout << "NetArch: IP:" << NumIPs << " H1:" << NumHN1 << " H2:" << NumHN2 << " OP:" << NumOPs << endl;
    cout << "Params: " << "LrnRate: " << LrnRate << "  Mtm1: " << Mtm1 << "  Mtm2: " << Mtm2 << endl;
    cout << endl;

    // Allocate memory for weights
//    w1 = Aloc2DAry(NumIPs, NumHN1); // 1st layer wts
//    w11 = Aloc2DAry(NumIPs, NumHN1);
//    w111 = Aloc2DAry(NumIPs, NumHN1);
//    w2 = Aloc2DAry(NumHN1, NumHN2); // 2nd layer wts
//    w22 = Aloc2DAry(NumHN1, NumHN2);
//    w222 = Aloc2DAry(NumHN1, NumHN2);
//    w3 = Aloc2DAry(NumHN2, NumOPs); // 3nd layer wts
//    w33 = Aloc2DAry(NumHN2, NumOPs);
//    w333 = Aloc2DAry(NumHN2, NumOPs);

    // Init wts between -0.5 and +0.5
    srand(time(0));
    for (i = 0; i < NumIPs; i++)
        for (j = 0; j < NumHN1; j++)
            w1[i][j] = w11[i][j] = w111[i][j] = float(rand()) / RAND_MAX - 0.5;
    for (i = 0; i < NumHN1; i++)
        for (j = 0; j < NumHN2; j++)
            w2[i][j] = w22[i][j] = w222[i][j] = float(rand()) / RAND_MAX - 0.5;
    for (i = 0; i < NumHN2; i++)
        for (j = 0; j < NumOPs; j++)
            w3[i][j] = w33[i][j] = w333[i][j] = float(rand()) / RAND_MAX - 0.5;
    // Initailize the array
    int *IndexLists = new int[NumPats];
    InitIndex(IndexLists, NumPats);
    if (Ordering >= 2)
        RandomSwap(IndexLists, NumPats);

    cout << "Training mlp for " << NumPats << " iterations:" << endl;
    cout << setprecision(6) << setw(7) << " #" << setw(12) << "MinErr" << setw(12) << "AveErr" << setw(12) << "MaxErr"
         << setw(12) << "%Err" << endl;

    for (;;) {
        MinErr = 3.4e38;
        AveErr = 0;
        MaxErr = -3.4e38;
        NumErr = 0;

        if (Ordering == 0) {
        } else if (Ordering == 1) {
            Randomize(IndexLists, NumPats);
        } else if (Ordering == 2) {
            RandomSwap(IndexLists, NumPats);
        } else {
            if (NumErr <= 1 / 10 * NumPats)
                Randomize(IndexLists, NumPats);
        }
        // Training
        for (int idx = 0; idx < NumPats; idx++) {
            p = IndexLists[idx];
            //forward propagation
            for (i = 0; i < NumHN1; i++) {
                float in = 0;
                for (j = 0; j < NumIPs; j++)
                    in += w1[j][i] * x[p][j];
                h1[i] = select_activate(activation, in); // Sigmoid fn
            }
            for (i = 0; i < NumHN2; i++) {
                float in = 0;
                for (j = 0; j < NumHN1; j++) {
                    in += w2[j][i] * h1[j];
                }
                h2[i] = select_activate(activation, in); // Sigmoid fn
            }
            for (i = 0; i < NumOPs; i++) {
                float in = 0;
                for (j = 0; j < NumHN2; j++) {
                    in += w3[j][i] * h2[j];
                }
                y[i] = select_activate(activation, in); // Sigmoid fn
            }
            // Cal error for this pattern
            PatErr = 0.0;
            for (i = 0; i < NumOPs; i++) {
                float err = y[i] - d[p][i]; // actual-desired O/P
                if (err > 0)
                    PatErr += err;
                else
                    PatErr -= err;
                NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) ||
                           (y[i] >= 0.5 && d[p][i] < 0.5)); //added for binary classification problem
            }
            if (PatErr < MinErr)
                MinErr = PatErr;
            if (PatErr > MaxErr)
                MaxErr = PatErr;
            AveErr += PatErr;

            for (i = 0; i < NumOPs; i++) {
                ad3[i] = (d[p][i] - y[i]) * y[i] * (1.0 - y[i]);
                for (j = 0; j < NumHN1; j++) {
                    w3[j][i] += LrnRate * h2[j] * ad3[i] +
                                Mtm1 * (w3[j][i] - w33[j][i]) +
                                Mtm2 * (w33[j][i] - w333[j][i]);
                    w333[j][i] = w33[j][i];
                    w33[j][i] = w3[j][i];
                }
            }

            for (i = 0; i < NumHN2; i++) {
                float err = 0.0;
                for (j = 0; j < NumOPs; j++)
                    err += ad3[j] * w3[i][j];
                ad2[i] = err * h2[i] * (1.0 - h2[i]);
                for (j = 0; j < NumHN1; j++) {
                    w2[j][i] += LrnRate * ad2[i] * h1[j] +
                                Mtm1 * (w2[j][i] - w22[j][i]) +
                                Mtm2 * (w22[j][i] - w222[j][i]);
                    w222[j][i] = w22[j][i];
                    w22[j][i] = w2[j][i];
                }
            }

            for (i = 0; i < NumHN1; i++) {
                float err = 0.0;
                for (j = 0; j < NumHN2; j++)
                    err += ad2[j] * w2[i][j];
                ad1[i] = err * h1[i] * (1.0 - h1[i]);
                for (j = 0; j < NumIPs; j++) {
                    w1[j][i] += LrnRate * ad1[i] * x[p][j] +
                                Mtm1 * (w1[j][i] - w11[j][i]) +
                                Mtm2 * (w11[j][i] - w111[j][i]);
                    w111[j][i] = w11[j][i];
                    w11[j][i] = w1[j][i];
                }
            }
        } // end for each pattern

        ItCnt++;
        AveErr /= NumPats;
        float PcntErr = NumErr / float(NumPats) * 100.0;
        cout.setf(ios::fixed | ios::showpoint);
        cout << setprecision(6) << setw(6) << ItCnt << ": " << setw(12) << MinErr << setw(12) << AveErr << setw(12)
             << MaxErr << setw(12) << PcntErr << endl;

        if ((AveErr <= ObjErr) || (ItCnt == NumIts))
            break;
    } // end main learning loop
    // Free memory
    delete[] IndexLists;
    delete[] h1;
    delete[] y;
    delete[] ad1;
    delete[] ad2;
}

void
TrainNet4(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
          float **w3, float **w33, float **w333, float **w4, float **w44, float **w444, int NumIPs, int NumOPs,
          int NumPats, int Ordering, int NumHN1, int NumHN2, int NumHN3, int activation) {
    float *h1 = new float[NumHN1];    // O/Ps of hidden layer 1
    float *h2 = new float[NumHN2];    // O/Ps of hidden layer 2
    float *h3 = new float[NumHN3];    // O/Ps of hidden layer 2
    float *y = new float[NumOPs];    // O/P of Net
    float *ad1 = new float[NumHN1]; // HN1 back prop errors
    float *ad2 = new float[NumHN2]; // HN2 back prop errors
    float *ad3 = new float[NumHN3]; // HN3 back prop errors
    float *ad4 = new float[NumOPs]; // O/P back prop errors
    float PatErr;                    // Absolute error sum of the pattern
    float MinErr;                    // Minimum epoch error
    float AveErr;                    // Aveage error in one epoch
    float MaxErr;                    // maximum epoch error

    int p, i, j;     // for loops indexes
    long ItCnt = 0;     // Iteration counter
    long NumErr = 0; // Error counter (added for spiral problem)

    cout << "NetArch: IP:" << NumIPs << " H1:" << NumHN1 << " H2:" << NumHN2 << " H3:" << NumHN3 << " OP:" << NumOPs
         << endl;
    cout << "Params: " << "LrnRate: " << LrnRate << "  Mtm1: " << Mtm1 << "  Mtm2: " << Mtm2 << endl;
    cout << endl;

    // Allocate memory for weights
//    w1 = Aloc2DAry(NumIPs, NumHN1); // 1st layer wts
//    w11 = Aloc2DAry(NumIPs, NumHN1);
//    w111 = Aloc2DAry(NumIPs, NumHN1);
//    w2 = Aloc2DAry(NumHN1, NumHN2); // 2nd layer wts
//    w22 = Aloc2DAry(NumHN1, NumHN2);
//    w222 = Aloc2DAry(NumHN1, NumHN2);
//    w3 = Aloc2DAry(NumHN2, NumHN3); // 3rd layer wts
//    w33 = Aloc2DAry(NumHN2, NumHN3);
//    w333 = Aloc2DAry(NumHN2, NumHN3);
//    w4 = Aloc2DAry(NumHN3, NumOPs); // 4th layer wts
//    w44 = Aloc2DAry(NumHN3, NumOPs);
//    w444 = Aloc2DAry(NumHN3, NumOPs);

    // Init wts between -0.5 and +0.5
    srand(time(0));
    for (i = 0; i < NumIPs; i++)
        for (j = 0; j < NumHN1; j++)
            w1[i][j] = w11[i][j] = w111[i][j] = float(rand()) / RAND_MAX - 0.5;
    for (i = 0; i < NumHN1; i++)
        for (j = 0; j < NumHN2; j++)
            w2[i][j] = w22[i][j] = w222[i][j] = float(rand()) / RAND_MAX - 0.5;
    for (i = 0; i < NumHN2; i++)
        for (j = 0; j < NumHN3; j++)
            w3[i][j] = w33[i][j] = w333[i][j] = float(rand()) / RAND_MAX - 0.5;
    for (i = 0; i < NumHN3; i++)
        for (j = 0; j < NumOPs; j++)
            w4[i][j] = w44[i][j] = w444[i][j] = float(rand()) / RAND_MAX - 0.5;
    // Initailize the array
    int *Indexlist = new int[NumPats];
    InitIndex(Indexlist, NumPats);
    if (Ordering >= 2)
        Randomize(Indexlist, NumPats);

    cout << "Training mlp for " << NumPats << " iterations:" << endl;
    cout << setprecision(6) << setw(7) << " #" << setw(12) << "MinErr" << setw(12) << "AveErr" << setw(12) << "MaxErr"
         << setw(12) << "%Err" << endl;

    for (;;) {
        MinErr = 3.4e38;
        AveErr = 0;
        MaxErr = -3.4e38;
        NumErr = 0;
        if (Ordering == 0) {
        } else if (Ordering == 1) {
            Randomize(Indexlist, NumPats);
        } else if (Ordering == 2) {
            RandomSwap(Indexlist, NumPats);
        } else {
            if (NumErr <= 1 / 10 * NumPats)
                Randomize(Indexlist, NumPats);
        }
        for (int idx = 0; idx < NumPats; idx++) {
            p = Indexlist[idx];
            //forward propagation
            for (i = 0; i < NumHN1; i++) {
                float in = 0;
                for (j = 0; j < NumIPs; j++)
                    in += w1[j][i] * x[p][j];
                h1[i] = select_activate(activation, in); // Sigmoid fn
            }
            for (i = 0; i < NumHN2; i++) {
                float in = 0;
                for (j = 0; j < NumHN1; j++) {
                    in += w2[j][i] * h1[j];
                }
                h2[i] = select_activate(activation, in); // Sigmoid fn
            }
            for (i = 0; i < NumHN3; i++) {
                float in = 0;
                for (j = 0; j < NumHN2; j++) {
                    in += w3[j][i] * h2[j];
                }
                h3[i] = select_activate(activation, in); // Sigmoid fn
            }
            for (i = 0; i < NumOPs; i++) {
                float in = 0;
                for (j = 0; j < NumHN3; j++) {
                    in += w4[j][i] * h3[j];
                }
                y[i] = select_activate(activation, in); // Sigmoid fn

            }
            // Cal error for this pattern
            PatErr = 0.0;
            for (i = 0; i < NumOPs; i++) {
                float err = y[i] - d[p][i]; // actual-desired O/P
                if (err > 0)
                    PatErr += err;
                else
                    PatErr -= err;
                NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) ||
                           (y[i] >= 0.5 && d[p][i] < 0.5)); //added for binary classification problem
            }
            if (PatErr < MinErr)
                MinErr = PatErr;
            if (PatErr > MaxErr)
                MaxErr = PatErr;
            AveErr += PatErr;

            for (i = 0; i < NumOPs; i++) {
                ad4[i] = (d[p][i] - y[i]) * y[i] * (1.0 - y[i]);
                for (j = 0; j < NumHN3; j++) {
                    w4[j][i] += LrnRate * h3[j] * ad4[i] +
                                Mtm1 * (w4[j][i] - w44[j][i]) +
                                Mtm2 * (w44[j][i] - w444[j][i]);
                    w444[j][i] = w44[j][i]; // The last last time weight
                    w44[j][i] = w4[j][i];    // the last time weight
                }
            }
            // 3rd hidden layer -> 2nd hidden layer : NumHN3 -> NumHN2
            for (i = 0; i < NumHN3; i++) {
                float err = 0.0;
                for (j = 0; j < NumOPs; j++)
                    err += ad4[j] * w4[i][j];
                ad3[i] = err * h3[i] * (1.0 - h3[i]);
                for (j = 0; j < NumHN2; j++) {
                    w3[j][i] += LrnRate * h2[j] * ad3[i] +
                                Mtm1 * (w3[j][i] - w33[j][i]) +
                                Mtm2 * (w33[j][i] - w333[j][i]);
                    w333[j][i] = w33[j][i]; // The last last time weight
                    w33[j][i] = w3[j][i];    // the last time weight
                }
            }
            // 3th hidden layer -> 2nd hidden layer : NumHN2 -> NumHN1
            for (i = 0; i < NumHN2; i++) {
                float err = 0.0;
                for (j = 0; j < NumHN3; j++)
                    err += ad3[j] * w3[i][j];
                ad2[i] = err * h2[i] * (1.0 - h2[i]);
                for (j = 0; j < NumHN1; j++) {
                    w2[j][i] += LrnRate * h1[j] * ad2[i] +
                                Mtm1 * (w2[j][i] - w22[j][i]) +
                                Mtm2 * (w22[j][i] - w222[j][i]);
                    w222[j][i] = w22[j][i];
                    w22[j][i] = w2[j][i];
                }
            }
            // 2nd hidden layer -> 1st hidden layer : HumHN1-> NumIPs
            for (i = 0; i < NumHN1; i++) {
                float err = 0.0;
                for (j = 0; j < NumHN2; j++)
                    err += ad2[j] * w2[i][j];
                ad1[i] = err * h1[i] * (1.0 - h1[i]);
                for (j = 0; j < NumIPs; j++) {
                    w1[j][i] += LrnRate * x[p][j] * ad1[i] +
                                Mtm1 * (w1[j][i] - w11[j][i]) +
                                Mtm2 * (w11[j][i] - w111[j][i]);
                    w111[j][i] = w11[j][i];
                    w11[j][i] = w1[j][i];
                }
            }
        } // end for each pattern

        ItCnt++;
        AveErr /= NumPats;
        float PcntErr = NumErr / float(NumPats) * 100.0;
        cout.setf(ios::fixed | ios::showpoint);
        cout << setprecision(6) << setw(6) << ItCnt << ": " << setw(12) << MinErr << setw(12) << AveErr << setw(12)
             << MaxErr << setw(12) << PcntErr << endl;

        if ((AveErr <= ObjErr) || (ItCnt == NumIts))
            break;
    } // end main learning loop
    // Free memory
    delete[] Indexlist;
    delete[] h1;
    delete[] y;
    delete[] ad1;
    delete[] ad2;
}


void
TestNet2(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222, int NumIPs, int NumOPs,
         int NumPats, string &outPath, int NumHN1,
         int activation) {
    float PatErr = 0;
    float MinErr = 0;
    float AveErr = 0;
    float MaxErr = 0;
    float *h1 = new float[NumHN1];

    float *y = new float[NumOPs];
    int p = 0, i, j;
    long ItCnt = 0;
    long NumErr = 0;

    cout << endl;
    cout << "Testing mlp for " << endl;
    cout << setprecision(6) << setw(7) << "  " << setw(12) << "MinErr" << setw(12) << "AveErr" << setw(12) << "MaxErr"
         << setw(12) << "%Err" << endl;

    ofstream outFile;
    outFile.open(outPath, ios::out); // 打开模式可省略
    for (p = 0; p < NumPats; p++) {

        for (i = 0; i < NumHN1; i++) {
            float in = 0;
            for (j = 0; j < NumIPs; j++)
                in += w1[j][i] * x[p][j];
            h1[i] = select_activate(activation, in);
        }
        for (i = 0; i < NumOPs; i++) {
            float in = 0;
            for (j = 0; j < NumHN1; j++) {
                in += w2[j][i] * h1[j];
            }
            y[i] = select_activate(activation, in);
            outFile << y[i] << endl;
        }

        PatErr = 0.0;
        for (i = 0; i < NumOPs; i++) {
            float err = y[i] - d[p][i];
            if (err > 0)
                PatErr += err;
            else
                PatErr -= err;
            NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) || (y[i] >= 0.5 && d[p][i] < 0.5));
        }
        if (PatErr < MinErr)
            MinErr = PatErr;
        if (PatErr > MaxErr)
            MaxErr = PatErr;
        AveErr += PatErr;

    }
    AveErr /= NumPats;
    float PcntErr = NumErr / float(NumPats) * 100.0;
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(6) << setw(7) << "  " << setw(12) << MinErr << setw(12) << AveErr << setw(12) << MaxErr
         << setw(12) << PcntErr << endl;



    //outFile << "Mike" << ',' << 18 << ',' << "paiting" << endl;
    //outFile << "Tom" << ',' << 25 << ',' << "football" << endl;
    //outFile << "Jack" << ',' << 21 << ',' << "music" << endl;
    outFile.close();
}

void TestNet3(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
              float **w3, float **w33, float **w333, int NumIPs, int NumOPs, int NumPats, string &outPath, int NumHN1, int NumHN2,
              int activation) {
    float PatErr = 0;
    float MinErr = 0;
    float AveErr = 0;
    float MaxErr = 0;
    float *h1 = new float[NumHN1];
    float *h2 = new float[NumHN2];
    float *y = new float[NumOPs];
    int p, i, j;
    long ItCnt = 0;
    long NumErr = 0;

    cout << endl;
    cout << "Testing mlp for " << endl;
    cout << setprecision(6) << setw(7) << "  " << setw(12) << "MinErr" << setw(12) << "AveErr" << setw(12) << "MaxErr"
         << setw(12) << "%Err" << endl;

    ofstream outFile;
    outFile.open(outPath, ios::out); // 打开模式可省略

    for (p = 0; p < NumPats; p++) {
        //forward propagation
        for (i = 0; i < NumHN1; i++) {
            float in = 0;
            for (j = 0; j < NumIPs; j++) {
                in += w1[j][i] * x[p][j];
            }
            h1[i] = select_activate(activation, in);
        }
        for (i = 0; i < NumHN2; i++) {
            float in = 0;
            for (j = 0; j < NumHN1; j++) {
                in += w2[j][i] * h1[j];
            }
            h2[i] = select_activate(activation, in);
        }
        for (i = 0; i < NumOPs; i++) {
            float in = 0;
            for (j = 0; j < NumHN2; j++) {
                in += w3[j][i] * h2[j];
            }
            y[i] = select_activate(activation, in);
            outFile << y[i] << endl;
        }

        PatErr = 0.0;
        for (i = 0; i < NumOPs; i++) {
            float err = y[i] - d[p][i];
            if (err > 0)
                PatErr += err;
            else
                PatErr -= err;
            NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) || (y[i] >= 0.5 && d[p][i] < 0.5));
        }
        if (PatErr < MinErr)
            MinErr = PatErr;
        if (PatErr > MaxErr)
            MaxErr = PatErr;
        AveErr += PatErr;

    }
    AveErr /= NumPats;
    float PcntErr = NumErr / float(NumPats) * 100.0;
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(6) << setw(7) << "  " << setw(12) << MinErr << setw(12) << AveErr << setw(12) << MaxErr
         << setw(12) << PcntErr << endl;
}

void
TestNet4(float **x, float **d, float **w1, float **w11, float **w111, float **w2, float **w22, float **w222,
         float **w3, float **w33, float **w333, float **w4, float **w44, float **w444, int NumIPs, int NumOPs,
         int NumPats, string &outPath, int NumHN1, int NumHN2, int NumHN3,
         int activation) {
    float PatErr = 0;
    float MinErr = 0;
    float AveErr = 0;
    float MaxErr = 0;
    float *h1 = new float[NumHN1];
    float *h2 = new float[NumHN2];
    float *h3 = new float[NumHN3];
    float *y = new float[NumOPs];
    int p, i, j;
    long ItCnt = 0;
    long NumErr = 0;

    cout << endl;
    cout << "Testing mlp for " << endl;
    cout << setprecision(6) << setw(7) << "  " << setw(12) << "MinErr" << setw(12) << "AveErr" << setw(12) << "MaxErr"
         << setw(12) << "%Err" << endl;
    ofstream outFile;
    outFile.open(outPath, ios::out); // 打开模式可省略

    for (p = 0; p < NumPats; p++) {
        for (i = 0; i < NumHN1; i++) {
            float in = 0;
            for (j = 0; j < NumIPs; j++) {
                in += w1[j][i] * x[p][j];
            }
            h1[i] = select_activate(activation, in);
        }
        for (i = 0; i < NumHN2; i++) {
            float in = 0;
            for (j = 0; j < NumHN1; j++) {
                in += w2[j][i] * h1[j];
            }
            h2[i] = select_activate(activation, in);
        }
        for (i = 0; i < NumHN3; i++) {
            float in = 0;
            for (j = 0; j < NumHN2; j++) {
                in += w3[j][i] * h2[j];
            }
            h3[i] = select_activate(activation, in);
        }
        for (i = 0; i < NumOPs; i++) {
            float in = 0;
            for (j = 0; j < NumHN3; j++) {
                in += w4[j][i] * h3[j];
            }
            y[i] = select_activate(activation, in);
            outFile << y[i] << endl;
        }

        PatErr = 0.0;
        for (i = 0; i < NumOPs; i++) {
            float err = y[i] - d[p][i];
            if (err > 0)
                PatErr += err;
            else
                PatErr -= err;
            NumErr += ((y[i] < 0.5 && d[p][i] >= 0.5) || (y[i] >= 0.5 && d[p][i] < 0.5));
        }
        if (PatErr < MinErr)
            MinErr = PatErr;
        if (PatErr > MaxErr)
            MaxErr = PatErr;
        AveErr += PatErr;

    }
    AveErr /= NumPats;
    float PcntErr = NumErr / float(NumPats) * 100.0;
    cout.setf(ios::fixed | ios::showpoint);
    cout << setprecision(6) << setw(7) << "  " << setw(12) << MinErr << setw(12) << AveErr << setw(12) << MaxErr
         << setw(12) << PcntErr << endl;
}


float **Aloc2DAry(int m, int n) {
    //Allocates memory for 2D array
    float **Ary2D = new float *[m];
    if (Ary2D == NULL) {
        cout << "No memory!\n";
        exit(1);
    }
    for (int i = 0; i < m; i++) {
        Ary2D[i] = new float[n];
        if (Ary2D[i] == NULL) {
            cout << "No memory!\n";
            exit(1);
        }
    }
    return Ary2D;
}


//float ***Aloc3DAry(int m, int n, int g) {
//    //Allocates memory for 3D array
//    float ***Ary3D = new float **[m];
//    if (Ary3D == NULL) {
//        cout << "No memory!\n";
//        exit(1);
//    }
//    for (int i = 0; i < m; i++) {
//        Ary3D[i] = Aloc2DAry(n, g);
//        if (Ary3D[i] == NULL) {
//            cout << "No memory!\n";
//            exit(1);
//        }
//    }
//    return Ary3D;
//}


void Free2DAry(float **Ary2D, int n) {
    for (int i = 0; i < n; i++)
        delete[] Ary2D[i];
    delete[] Ary2D;
}

//void Free3DAry(float ***Ary3D, int m, int n) {
//    for (int i = 0; i < m; i++)
//        Free2DAry(Ary3D[i], n);
//    delete[] Ary3D;
//}

int CountLines(const char *filename) {
    ifstream ReadFile;
    int n = 0;
    string tmp;
    ReadFile.open(filename, ios::in);//ios::in 表示以只读的方式读取文件
    if (ReadFile.fail())//文件打开失败:返回0
    {
        return 0;
    }
    //文件存在
    while (getline(ReadFile, tmp, '\n')) {
        n++;
    }
    ReadFile.close();
    return n;
}


void
TrainOneState(string dataset, string prefix, string suffix, int NumHN, int NumHN1, int NumHN2, int NumHN3,
              int activation) {
    ifstream fin;
    char Line[500];
    string inFile = prefix + dataset + suffix;
    int NumTrnPats = CountLines(inFile.c_str()) / 2;
    cout << NumTrnPats << endl;
    int NumTstPats = NumTrnPats;
    cout << inFile << endl;
    fin.open(inFile.c_str(), ios::in);
    if (!fin.good()) {
        cout << "File not found!\n";
        exit(1);
    }
    fin.getline(Line, 500);

    if (NumIPs < 1 || NumIPs > MAXN || NumOPs < 1 || NumOPs > MAXN ||
        NumTrnPats < 1 || NumTrnPats > MAXPATS || NumTrnPats < 1 || NumTrnPats > MAXPATS ||
        NumIts < 1 || NumIts > 20e6 || NumHN1 < 0 || NumHN1 > 50 ||
        LrnRate < 0 || LrnRate > 1 || Mtm1 < 0 || Mtm1 > 10 || Mtm2 < 0 || Mtm2 > 10 || ObjErr < 0 || ObjErr > 10) {
        cout << "Invalid specs in data file!\n";
        exit(1);
    }

    float **IPTrnData = Aloc2DAry(NumTrnPats, NumIPs);
    float **OPTrnData = Aloc2DAry(NumTrnPats, NumOPs);
    float **IPTstData = Aloc2DAry(NumTstPats, NumIPs);
    float **OPTstData = Aloc2DAry(NumTstPats, NumOPs);

    for (int i = 0; i < NumTrnPats; i++) {
        for (int j = 0; j < NumIPs; j++)
            fin >> IPTrnData[i][j];
        for (int j = 0; j < NumOPs; j++)
            fin >> OPTrnData[i][j];
    }

    for (int i = 0; i < NumTstPats; i++) {
        for (int j = 0; j < NumIPs; j++)
            fin >> IPTstData[i][j];
        for (int j = 0; j < NumOPs; j++)
            fin >> OPTstData[i][j];
    }
    fin.close();

    // 11表示字符串尾部".normal.csv"的长度
//    string outPath = inPath.substr(0, inPath.length() - 11) + "_" + to_string(NumHN) + "_" + to_string(NumHN1) +
//                     (NumHN2 != 0 ? ("_" + to_string(NumHN2)) : "") + (NumHN3 != 0 ? ("_" + to_string(NumHN3)) : "") +
//                     ".csv";
    string folder = activation ? "guass1" : "sigmoid";
    string outPath =
            prefix + folder + "/" + dataset + "/" + dataset + "_" + to_string(NumHN) + "_" + to_string(NumHN1) +
            (NumHN2 != 0 ? ("_" + to_string(NumHN2)) : "") + (NumHN3 != 0 ? ("_" + to_string(NumHN3)) : "") +
            ".csv";
    cout << outPath << endl;

    // mlp weights
    float **w1, **w11, **w111; // 1st layer wts
    float **w2, **w22, **w222; // 2nd layer wts
    float **w3, **w33, **w333; // 3rd layer wts
    float **w4, **w44, **w444; // 4th layer wts
    float **w5, **w55, **w555; // 5th layer wts

    // Allocate memory for weights
    w1 = Aloc2DAry(NumIPs, NumHN1); // 1st layer wts
    w11 = Aloc2DAry(NumIPs, NumHN1);
    w111 = Aloc2DAry(NumIPs, NumHN1);
    w2 = Aloc2DAry(NumHN1, NumHN2); // 2nd layer wts
    w22 = Aloc2DAry(NumHN1, NumHN2);
    w222 = Aloc2DAry(NumHN1, NumHN2);
    w3 = Aloc2DAry(NumHN2, NumHN3); // 3rd layer wts
    w33 = Aloc2DAry(NumHN2, NumHN3);
    w333 = Aloc2DAry(NumHN2, NumHN3);
    w4 = Aloc2DAry(NumHN3, NumOPs); // 4th layer wts
    w44 = Aloc2DAry(NumHN3, NumOPs);
    w444 = Aloc2DAry(NumHN3, NumOPs);

    switch (NumHN) {
        case 3:
            TrainNet4(IPTrnData, OPTrnData, w1, w11, w111, w2, w22, w222, w3, w33, w333, w4, w44, w444, NumIPs, NumOPs,
                      NumTrnPats, Ordering, NumHN1,
                      NumHN2, NumHN3, activation);
            TestNet4(IPTstData, OPTstData, w1, w11, w111, w2, w22, w222, w3, w33, w333, w4, w44, w444, NumIPs, NumOPs,
                     NumTstPats, outPath, NumHN1,
                     NumHN2, NumHN3, activation);
            break;
        case 2:
            TrainNet3(IPTrnData, OPTrnData, w1, w11, w111, w2, w22, w222, w3, w33, w333, NumIPs, NumOPs, NumTrnPats,
                      Ordering, NumHN1, NumHN2, activation);
            TestNet3(IPTstData, OPTstData, w1, w11, w111, w2, w22, w222, w3, w33, w333, NumIPs, NumOPs, NumTstPats,
                     outPath, NumHN1, NumHN2, activation);
            break;
        default:
            TrainNet2(IPTrnData, OPTrnData, w1, w11, w111, w2, w22, w222, NumIPs, NumOPs, NumTrnPats, Ordering, NumHN1,
                      activation);
            TestNet2(IPTstData, OPTstData, w1, w11, w111, w2, w22, w222, NumIPs, NumOPs, NumTstPats, outPath, NumHN1,
                     activation);
    }

    Free2DAry(IPTrnData, NumTrnPats);
    Free2DAry(OPTrnData, NumTrnPats);
    Free2DAry(IPTstData, NumTstPats);
    Free2DAry(OPTstData, NumTstPats);
    cout << "End of program.\n";
    //system("PAUSE"); // win32
    //int c = getchar(); // alternative to system("PAUSE") in UNIX
    return;
}

double gauss1(double x, double c, double sig) {
    double a = pow((x - c), 2);
    double b = -a / (2 * sig);
    double y = exp(b);
    return y;
}

double gauss2(double x, double a, double b, double r) {
    double t1 = a * x - b;
    double t2 = pow(-t1, r);
    double y = exp(t2);
//    cout << x << " " << a << " " << b << " " << r << endl;
//    cout << t1 << " " << t2 << " " << y << endl;
    return y;
}

double sigmoid(double x) {
    double y = (double) (1.0 / (1.0 + exp(double(-x))));
    return y;
}

int order(int Ordering, int Indexlist, int NumPats, int NumErr) {
    if (Ordering == 0) {
    } else if (Ordering == 1) {
        Randomize(&Indexlist, NumPats);
    } else if (Ordering == 2) {
        RandomSwap(&Indexlist, NumPats);
    } else {
        if (NumErr <= 1 / 10 * NumPats)
            Randomize(&Indexlist, NumPats);
    }
    return 0;
}

double select_activate(int order, double x) {
    double y = 0;
    if (order == 0) {
        y = sigmoid(x);
    } else if (order == 1) {
        y = gauss1(x, 0.5, 100);
    } else if (order == 2) {
        y = gauss2(x, 1, 100000000000, 0.0000005);
    }
    return y;
}


void createDir(string path) {
    FILE *fp = NULL;
    fp = fopen(path.c_str(), "w");

    if (!fp) {
        mkdir(path.c_str(), 0775);
    } else {
        fclose(fp);
    }
}
