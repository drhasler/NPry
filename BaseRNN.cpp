// still bad

#pragma once
using namespace std;
//#include "RNN_tok_friendly.hpp"
#include<bits/stdc++.h>
#include "Model.cpp"
#define FOR(a,b) for(int a=0;a<b;a++)
#define ROF(a,b) for(int a=b-1;a--;)
#define vi vector<double>
#define vd vector<double>
#define mat vector<vd>


class BaseRNN : public Model {
public:
    int HS,VS;
    mat Wxh , // input to hidden
    Whh , // hidden to hidden
    Why; //  hidden to output
    vd bh,
    by;
    vector<vector<double>> m1,m2,m3;
    
    BaseRNN() : Model(_baseRNN) {
        string ans;
        
        cout << "creating new BaseRNN<";
        cout << ">:\ninput max VS dim (default 128): " << flush;
        
        do { getline(cin,ans);
            VS = atoi(ans.c_str()); }
        
        
        
        while (VS > 8000 or VS < 2); {
            getline(cin,ans);
            VS = atoi(ans.c_str());
            if (VS<2) {
                cout << "too small - size must be between 2 and 8000\n";
            }
            
            if (VS>8000) {
                cout << "too big - size must be between 2 and 8000\n";
            }
            
        }
        cout << "voc size set to " << VS << endl;
        //build_tok();
        
        cout << "input max HS dim (default 128): " << flush;
        getline(cin,ans);
        HS = atoi(ans.c_str());
        if (HS<2) HS = 128;
        if (HS>8000) { // or check if memset is happy
            cout << "too bigg\n";
            HS = 128;
        }
        cout << "hidden size set to " << HS << endl;
        // construct matrices
        Wxh =  mat(HS,vd(VS,0)), // input to hidden
        Whh =  mat(HS,vd(HS,0)), // hidden to hidden
        Why =  mat(VS,vd(HS,0)); //  hidden to output
        vd bh =  vd(HS,0),
        by =  vd(VS,0);
        resetWm();
    }
    
    
    map<string,int> occurences;
    map<string,int> s_to_tok;
    map<int,string> tok_to_s;
    const int seq_length  = 25; // number of steps to unroll the RNN for
    const double learning_rate = 1e-1;
    
    void reset(string& w,char c,int& state) {
        w = string(1,c);
        state = isalpha(c)?0:isspace(c)?1:ispunct(c)?2:3;
    }
    
    vector<string> parse(const string &line) {
        vector<string> ans;
        
        string w; // cur word
        int state = 0;
        for (char c:line) {
            c = tolower(c);
            if (state==0) { // abc sequence
                if (isalpha(c)) {
                    w += c;
                } else {
                    ans.push_back(w);
                    reset(w,c,state);
                }
            } else if (state==1) { // whitespaces
                reset(w,c,state);
            } else if (state==2) { // punctuation
                ans.push_back(w);
                reset(w,c,state);
            } else reset(w,c,state); // numbers and weird chars
        }
        if (state==0 || state==2) ans.push_back(w);
        ans.push_back("\n");
        return ans;
    }
    
    void add_occurences(string filename) {
        ifstream ifs(filename);
        string line;
        while(getline(ifs,line)) {
            vector <string> parsed = parse(line);// for each word or ponctuation mark and \n
            for (auto w:parsed) occurences[w]++;
        }
    }
    
    void build_dic(int& voc_size) {
        voc_size--;
        vector<pair<string,int> > temp;
        for (auto x:occurences) temp.push_back(x);
        if (temp.size()<voc_size) voc_size = temp.size();
        // puts all elements leq than the nth one in temp[:n]
        // and the rest in temp[n:]
        nth_element(temp.begin(),temp.begin()+voc_size,temp.end(),
                    [&](auto a,auto b){return a.second>b.second;});
        for (int i=0;i<voc_size;i++) {
            s_to_tok[temp[i].first] = i+1;
            tok_to_s[i+1] = temp[i].first;
        }
        voc_size++;
        cout << (temp.size()-voc_size) << " untokenized words \n";
        auto sumry = [](int a,pair<string,int> b) {return a+b.second;};
        double untok_prop = accumulate(temp.begin()+voc_size,temp.end(),0,sumry)*100.0/
        accumulate(temp.begin(),temp.end(),0,sumry);
        cout << untok_prop << " % hasn't been tokenized \n";
    }
    
    vector<int> tokenize(string filename) {
        vector<int> tokenized;
        ifstream ifs(filename);
        string line;
        while(getline(ifs,line)) {
            vector <string> parsed = parse(line);// for each word or ponctuation mark and \n
            for (auto w:parsed){
                if (s_to_tok.count(w)){
                    tokenized.push_back(s_to_tok[w]);
                }
                else {
                    tokenized.push_back(0);
                }
            }
        }
        return tokenized;
    }
    
    
    void resetWm(){
        random_device rd{};
        mt19937 gen{rd()};
        normal_distribution<> d{0,1};
        for (auto W: {&Wxh,&Whh,&Why})
            for_each(W->begin(),W->end(),
                     [&](vd& row){std::generate(row.begin(),row.end(),
                                                [&](){
                                                    return d(gen)*.01;} ); });
        cout << "model generated\n";
    }
    
    /*mat  *Wxh = new mat(HS,vd(VS,0)), // input to hidden
     *Whh = new mat(HS,vd(HS,0)), // hidden to hidden
     *Why = new mat(VS,vd(HS,0)); //  hidden to output
     vd *bh = new vd(HS,0),
     *by = new vd(VS,0);*/
    
    tuple<double,mat,mat,mat,vd,vd,vd> lossFun
    (const vi& inputs, const vi& targets, const vd& hprev) {
        mat xs(seq_length,vd(VS,0)),// 1hot input
        hs(seq_length,vd(HS,0)),  // hidden state
        ys(seq_length,vd(VS,0)),  // score
        ps(seq_length,vd(VS,0));  // proba
        double loss = 0;
        FOR(t,seq_length) { // forward pass
            xs[t][inputs[t]] = 1;
            FOR(i,HS) {
                hs[t][i] = Wxh[i][inputs[t]] + bh[i];
                FOR(j,HS) hs[t][i] += Whh[i][j] * (t ? hs[t-1][j] : hprev[j]);
                hs[t][i] = tanh(hs[t][i]);
            }
            double Y = 0;
            FOR(i,VS) {
                ys[t][i] = by[i];
                FOR(j,HS) ys[t][i] += Why[i][j] * hs[t][j];
                Y += exp(ys[t][i]);
            }
            FOR(i,VS) ps[t][i] = exp(ys[t][i])/Y;
            loss -= log(ps[t][targets[t]]);
        }
        mat dWxh (HS,vd(VS,0));
        mat dWhh (HS,vd(HS,0));
        mat dWhy (VS,vd(HS,0));
        vd dbh (HS,0);
        vd dby (VS,0);
        vd dhnex (HS,0);
        ROF(t,seq_length) { // backward pass
            vd dy = ps[t];
            dy[targets[t]] -= 1.0;
            FOR(i,VS) FOR(j,HS)
            dWhy[i][j] += dy[i]*hs[t][j];
            FOR(i,VS) dby[i] += dy[i];
            vd dh = dhnex;
            FOR(j,VS) FOR(i,HS)
            dh[i] += Why[j][i]*dy[j];
            FOR(i,HS) dh[i] *= (1-hs[t][i]*hs[t][i]); // dh is now dhraw
            FOR(i,HS) dbh[i] += dh[i];
            FOR(i,HS) dWxh[i][inputs[t]] += dh[i];
            FOR(i,HS) FOR(j,HS)
            dWhh[i][j] += dh[i]*(t?hs[t-1][j]:hprev[j]);
            fill(dhnex.begin(),dhnex.end(),0);
            FOR(i,HS) FOR(j,HS)
            dhnex[j] += Whh[i][j]*dh[i];
        }
        for (auto dW:{dWxh,dWhh,dWhy})
            for_each(dW.begin(),dW.end(),
                     [](vd& row){transform(row.begin(),row.end(),row.begin(),
                                           [](double val){ return min(5.0,max(-5.0,val)); }); });
        for (auto db:{dbh,dby})
            transform(db.begin(),db.end(),db.begin(),
                      [](double val){ return min(5.0,max(-5.0,val)); });
        return {loss,dWxh,dWhh,dWhy,dbh,dby,hs[seq_length-1]};
    }
    
//=================Interesting functions ======================    
    
    vi gen(vi seed, int l) { // generate sequence of length l
        vd h(HS, 0); 
        int x = 0;
        vi ans(l);
        vd h_aux (HS),p(VS);
        FOR(t,l) {
            fill(h_aux.begin(),h_aux.end(),0);
            FOR(i,HS) FOR(j,HS) h_aux[i] += Whh[i][j] * h[j];
            FOR(i,HS) h[i] = tanh(Wxh[i][x] + h_aux[i] + bh[i]);
            auto y = by;
            assert(Why.size()==VS && Why[0].size()==HS);
            FOR(i,VS) FOR(j,HS) y[i] += Why[i][j]*h[j];
            double bigY = 0;
            for_each(y.begin(),y.end(),[&](double v){bigY += exp(v);});
            transform(y.begin(),y.end(),p.begin(),[=](double v){return exp(v)/bigY;});
            discrete_distribution<> prob_d(p.begin(),p.end());
            random_device rd{};
            mt19937 gen{rd()};
            ans[t] = x = prob_d(gen);
        }
        return ans;
    }
    
    
    void train(vi inputs, vi outputs) {
        vi data;
        string filename = "../training_texts/hakuk.txt";
        
        add_occurences(filename);
        
        VS = 3000;
        build_dic(VS);
        tok_to_s[0] = "<unk>";
        
        int n = 0,
        // nice memories
        mat mWxh (HS,vd(VS,0)),
        	mWhh (HS,vd(HS,0)),
        	mWhy (VS,vd(HS,0));
        vd  mbh (HS,0),
        	mby (VS,0),
        	hprev (HS,0);
        double smooth_loss = -log(1.0/VS)*seq_length;
        while (1) {
            if (p > p_max) {
                fill(hprev.begin(),hprev.end(),0);
                p = rand() % seq_length;
            }
            
            FOR(i, seq_length) inputs[i] = data [p+i];
            FOR(i, seq_length) targets[i] = data [p+i+1];
            
            if (n % 300 == 0) {
                vi sample_ix = sample(hprev,inputs[0],50);
                printf("--- iter %d sample ---\n",n);
                for (int i:sample_ix) cout << tok_to_s[i]<<' ';
                cout << "\n- - - - - - - - - - - - - - - -\n";
            }
            // feed sequence and get gradient
            // tuple<double,mat,mat,mat,vd,vd,vd> lossFun
            double loss;
            mat dWxh;
            mat dWhh;
            mat dWhy;
            vd dbh;
            vd dby;
            vd hp;
            
            tie(loss, dWxh, dWhh, dWhy, dbh, dby, hp) = lossFun(inputs, targets, hprev);
            
            hprev = hp;
            smooth_loss = smooth_loss * 0.999 + loss * 0.001;
            if (n % 100 == 0) printf("iter %d, loss: %f\n",n,smooth_loss);
            //color that^ line with ../Uinterface/goodness_color.cpp 's coloredLossString() to which you can pass loss with a small operation applied to it
            //task4u
            
            
            // update with Adagrad
            for (auto X : { tuple<mat*,mat*,mat*>
                ({&Wxh,&dWxh,&mWxh}),
                {&Whh,&dWhh,&mWhh},
                {&Why,&dWhy,&mWhy}}) {
                    mat &W = *get<0>(X),
                    &dW = *get<1>(X),
                    &mem = *get<2>(X);
                    FOR(i,(int)W.size()) FOR(j,(int)W[0].size()) {
                        mem[i][j] += dW[i][j] * dW[i][j];
                        W[i][j] -= learning_rate * dW[i][j] / sqrt(mem[i][j] + 1e-8);
                    }
                }
            for (auto X : {tuple<vd*,vd*,vd*>
                ({&bh,&dbh,&mbh}),
                {&by,&dby,&mby}}) {
                    vd &b = *get<0>(X),
                    &db = *get<1>(X),
                    &mem = *get<2>(X);
                    FOR(i,(int)b.size()) {
                        mem[i] +=db[i] * db[i];
                        b[i] -= learning_rate * db[i] / sqrt(mem[i] + 1e-8);
                    }
                }
            p += seq_length;
            n++;
        }

    }
#define RC(x) reinterpret_cast<char*>(&x),sizeof(x)
    void save(string filename) {
    	mType ty;
        ofstream ofs(filename,ios::binary);
        ofs.write(RC(ty));
        ofs.write(RC(HS));
        ofs.write(RC(VS));
        for (auto& v:m1) for (auto x:v) ofs.write(RC(x));
        for (auto& v:m2) for (auto x:v) ofs.write(RC(x));
        for (auto& v:m3) for (auto x:v) ofs.write(RC(x));
    }
    
    void load(string filename) {
    	mType ty;
        ifstream ifs(filename,ios::binary);
        ifs.read(RC(ty));
        ifs.read(RC(HS));
        ifs.read(RC(VS));
        for (auto& v:m1) for (auto& x:v) ifs.read(RC(x));
        for (auto& v:m2) for (auto& x:v) ifs.read(RC(x));
        for (auto& v:m3) for (auto& x:v) ifs.read(RC(x));
    }
#undef RC
    
    
};

#undef FOR

