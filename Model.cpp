#pragma once
#include <bits/stdc++.h>
#include "Piper.cpp"
#include <Eigen/Eigen>
using namespace std;
enum modelType {_baseRNN,_eigRNN,_eigLSTM,_Ngram};

using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
class Param {
public:
    mat v,d,m;
    Param() {}
    Param(int in, int out=1) {
        v = mat::Zero(in,out); // value
        d = mat::Zero(in,out); // derivative
        m = mat::Zero(in,out); // memory
    }
};


// models only deal with tokens. IO handled by some wrapper
class Model {
public:
    // always constructed with a type
    const modelType mt;
    double loss;
    int VS;
    bool b=0;
    // handles the IO, tokens etc
    Piper* p;

    Model(modelType MT,Piper* x) : mt(MT),p(x) { }

    ~Model() { delete p; }

    string strgen(int len, string seed) {
        vector<int> vseed = p->tokenize(seed);
        vector<int> vgen  = gen(vseed, len, b);
        return p->untokenize(vgen);
    }
    void textgen(int len, string seed = "") { cout << strgen(len,seed) << endl; }
    void filegen(string filename,int len,string seed = "") {
        ofstream ofs(filename);
        ofs << strgen(len,seed);
    }
    void texttrain(string& text) {
        vector<int> vin = p->tokenize(text);
        vector<int> vout(vin.begin()+1, vin.end());
        vin.pop_back();
        train(vin,vout);
    }
    // pure virtual calls
    virtual vector<int> gen(vector<int> seed, int len, bool rejecc_unk) = 0;
    virtual void train(vector<int> inputs, vector<int> outputs) = 0;
    virtual void bigLearn(vector<int> data,int batch_size=25,int epoch=30) = 0;
    // you have to retokenize the same before saving or loading
    virtual void load(string) = 0;
    virtual void save(string) = 0;
};
