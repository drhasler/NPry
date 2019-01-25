// based on python code from Andrej Karpathy (@karpathy)
// the director of artificial intelligence and Autopilot Vision at Tesla
// If you see this: Runtime errors due to Matrices are fixed
#include <bits/stdc++.h>
#include <Eigen/Eigen>
#include "Model.cpp"
using namespace std;
#define FOR(a,b) for(int a=0;a<b;a++)
#define ROF(a,b) for(int a=b-1;a--;)
#define T    transpose()
#define A    array()
#define mat  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> 
#define vec  Eigen::Matrix<double,Eigen::Dynamic,1> 
#define vint vector<int>


class eigRNN : public Model {
public:

    int HS,XS; 
    Param Wxh,Whh,Why,bh,by;
    vec h;

    double learning_rate = 1e-1;
    double smooth_loss;

    eigRNN(Piper* p) : Model(_eigRNN,p), XS(p->unique_wc) {
        cout << "created eigRNN" << p->repr << endl;
        HS = safe_int("hidden layer size",1,1000);
        // 0 initialization
        Wxh = Param(HS,XS), // input  to hidden
        Whh = Param(HS,HS), // hidden to hidden
        Why = Param(XS,HS); // hidden to output
        bh = Param(HS),
        by = Param(XS); 
        h = vec::Zero(HS); 
        { // fill matrices
            random_device rd{};
            mt19937 gen(rd());
            normal_distribution<> norm_d{0,1};
            for (auto W:{&Wxh,&Whh,&Why})
                W->v = W->v.unaryExpr([&](double x){return .01 * norm_d(gen);});
        }
        cout << "successful init" << endl;
    }

    inline void clear_grad() {
        for (auto p:{&Wxh,&Whh,&bh,&Why,&by})
            p->d.setZero();
    }

    inline void clip_grad() {
        for (auto p:{&Wxh,&Whh,&bh,&Why,&by})
            p->d = p->d.A.min(5.0).max(-5.0);
    }

    void bigLearn(const vint data,const int SL,const int epochs) {
        double init_loss = -log(1.0/XS)*SL;
        smooth_loss = init_loss;
        int imax = data.size() - SL - 1;
        mat xs(XS,SL),  // 1hot input
            hs(HS,SL+1),// hidden state
            ys(XS,SL),  // score
            ps(XS,SL);  // proba

        printf("will run %d batches of size %d over %d epochs\n",imax/SL,SL,epochs);
        cout << smooth_loss << endl;
            random_device rd{};
            mt19937 gen(rd());

        G_signal = 0;
        FOR(e,epochs) {
            h.setZero();
            for (int i=gen()%SL;i<imax;i+=SL) {
            if (G_signal) {
                cout << "\nprocess interupted"<<endl;
                if (safe_choice("would you like to stop? ")) return;
                else G_signal = 0;
            }
            xs.setZero();
            hs.col(0) = h;
            double loss = 0;

            FOR(t,SL) { // forward pass
                xs(data[t+i],t) = 1.0; // hott
                hs.col(t+1) = tanh((Wxh.v*xs.col(t) + Whh.v*hs.col(t) + bh.v).A); // pull
                ys.col(t) = Why.v*hs.col(t+1) + by.v; // putt
                ps.col(t) = exp(ys.col(t).A)/exp(ys.col(t).A).sum(); // soft
                loss -= log(ps(data[t+i+1],t));
            }

            clear_grad();

            vec dhnext = vec::Zero(HS);
            ROF(t,SL) { // backward pass
                vec dy = ps.col(t);
                dy(data[t+i+1]) -= 1.0;
                // out
                Why.d += dy*hs.col(t+1).T;
                by.d  += dy;

                vec dh = Why.v.T*dy + dhnext;
                vec dhraw = (1.0-hs.col(t+1).A.square())*dh.A;
                // hidden state
                bh.d += dhraw;
                Whh.d += dhraw*hs.col(t).T;
                // input layer
                Wxh.d += dhraw*xs.col(t).T;
                dhnext = Whh.v.T*dhraw;
            }
            clip_grad();
            // update
            h = hs.col(SL);

            smooth_loss = .999*smooth_loss + .001*loss;

            for (auto p:{&Wxh,&Whh,&bh,&Why,&by}) {
                p->m.A += p->d.A*p->d.A;
                p->v.A -= learning_rate * p->d.A / sqrt(p->m.A + 1e-8);
            }

                reset_line();
                set_good((110.0-smooth_loss)/70.0);
                printf("loss: %.2f - epoch %d progression %.2f",smooth_loss,e,i/(double)imax);
                fflush(stdout);
            if (i==-1) {
                vec HH = h;
                if (i%1000==0) { textgen(300); }
                h = HH;
            }
        } }
        cout << "\ndone" << endl;
    }
    
    vector<int> gen(vector<int> seed, int SL, bool rejecc_unk=0){
        random_device zak{};
        mt19937 g(zak());
        vint seq(SL);
        vec x(XS),y(XS),p(XS);
        for (int s : seed){
            x = vec::Zero(XS);
            x(s) = 1.0;
            h = tanh((Wxh.v*x + Whh.v*h + bh.v).A);
            y = Why.v*h + by.v;
            p = exp(y.A) / exp(y.A).sum();
            discrete_distribution<> prob_d(p.data(),p.data()+XS);
            seq[0] = prob_d(g);
            x = vec::Zero(XS); x(seq[0]) = 1.0;
        }

        FOR(t,SL) {
            h = tanh((Wxh.v*x + Whh.v*h + bh.v).A);
            y = Why.v*h + by.v;
            p = exp(y.A) / exp(y.A).sum();
            discrete_distribution<> prob_d(p.data(),p.data()+XS);
            seq[t] = prob_d(g);
            x = vec::Zero(XS); x(seq[t]) = 1.0;
        }
        return seq;
    }

    #define RC(x) reinterpret_cast<char*>(&x),sizeof(x)
    void save(string fname) {
        ofstream ofs(fname,ios::binary);
        ofs.write(RC(HS));
        ofs.write(RC(XS));
        for (auto p:{&Wxh, &Whh, &bh, &Why, &by}) {
            int h = p->m.rows(), w = p->m.cols();
            ofs.write(RC(h));
            ofs.write(RC(w));
            FOR(i,h*w) ofs.write(RC(*(p->v.data()+i)));
            FOR(i,h*w) ofs.write(RC(*(p->m.data()+i)));
        }
        ofs.close();
    }

    void load(string fname){
        ifstream ifs(fname,ios::binary);
        ifs.read(RC(HS));
        ifs.read(RC(XS));
        for (auto p:{&Wxh, &Whh, &bh, &Why, &by}) {
            int h,w;
            ifs.read(RC(h));
            ifs.read(RC(w));
            p->v.resize(h,w);
            p->m.resize(h,w);
            FOR(i,h*w) ifs.read(RC(*(p->v.data()+i)));
            FOR(i,h*w) ifs.read(RC(*(p->m.data()+i)));
        }
        ifs.close();
    }
    #undef RC //we don't want KONFLIKT
};

#undef FOR 
#undef ROF 
#undef T   
#undef A   
#undef mat 
#undef vec 
#undef vint
