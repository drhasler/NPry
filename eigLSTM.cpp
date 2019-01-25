#pragma once
#include <bits/stdc++.h>
#include <Eigen/Eigen>
#include "Model.cpp"
#include "../utils/IO.cpp"
using namespace std;
#define FOR(a,b) for(int a=0;a<b;a++)
#define ROF(a,b) for(int a=b-1;a--;)
#define T transpose()
#define A array()

#define vint std::vector<int>
#define mat  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
#define vec  Eigen::Matrix<double,Eigen::Dynamic,1>

#define dim(x) printf("[%d,%d] x\n",x.rows(),x.cols())

class eigLSTM : public Model { 
public:
    // weights,biases,activation
    Param Wf,bf,//sig
          Wi,bi,//tanh
          Wo,bo,//sig
          WC,bC,//sig
          Wv,bv;//tanh
    int HS,ZS,XS,IT;
    vec h,C;

    double smooth_loss, learning_rate = 1e-1;
    eigLSTM(Piper* p) : Model(_eigLSTM, p), XS(p->unique_wc) {
        HS = safe_int("hidden layer size",1,1000);
        ZS = XS + HS;
        // 0 initialization
        Wf = Param(HS,ZS), bf = Param(HS,1);
        Wi = Param(HS,ZS), bi = Param(HS,1);
        Wo = Param(HS,ZS), bo = Param(HS,1);
        WC = Param(HS,ZS), bC = Param(HS,1);
        Wv = Param(XS,HS), bv = Param(XS,1);
        h = vec::Zero(HS), C = vec::Zero(HS);
        { // fill matrices
            random_device rd{};
            mt19937 gen(rd());
            normal_distribution<> norm_d{0,1};
            for (auto W:{&Wf,&Wi,&Wo})
                W->v = W->v.unaryExpr([&](double x){return .5 + .01 * norm_d(gen);});
            for (auto W:{&WC,&Wv})
                W->v = W->v.unaryExpr([&](double x){return .01 * norm_d(gen);});
            smooth_loss = -log(1.0/XS)*25;
        }
    }

    inline void clear_grad() {
        for (auto p:{&Wf,&bf,&Wi,&bi,&Wo,&bo,&WC,&bC,&Wv,&bv}) {
            p->d.setZero();
        }
    }

    inline void clip_grad() {
        for (auto p:{&Wf,&bf,&Wi,&bi,&Wo,&bo,&WC,&bC,&Wv,&bv}) {
            p->d = p->d.A.min(5.0).max(-5.0);
        }
    }


#define RC(x) reinterpret_cast<char*>(&x),sizeof(x)

    void save(string fname) {
        ofstream ofs(fname,ios::binary);
        ofs.write(RC(HS));
        ofs.write(RC(ZS));
        ofs.write(RC(XS));
        for (auto p:{&Wf,&bf,&Wi,&bi,&Wo,&bo,&WC,&bC,&Wv,&bv}) {
            int h = p->m.rows(), w = p->m.cols();
            ofs.write(RC(h));
            ofs.write(RC(w));
            FOR(i,h*w) ofs.write(RC(*(p->v.data()+i)));
            FOR(i,h*w) ofs.write(RC(*(p->m.data()+i)));
        }
        ofs.close();
    }

    void load(string fname) {
        ifstream ifs(fname,ios::binary);
        ifs.read(RC(HS));
        ifs.read(RC(ZS));
        ifs.read(RC(XS));
        for (auto p:{&Wf,&bf,&Wi,&bi,&Wo,&bo,&WC,&bC,&Wv,&bv}) {
            int h,w;
            ifs.read(RC(h));
            ifs.read(RC(w));
            p->v.resize(h,w);
            p->d.resize(h,w);
            p->m.resize(h,w);
            FOR(i,h*w) ifs.read(RC(*(p->v.data()+i)));
            FOR(i,h*w) ifs.read(RC(*(p->m.data()+i)));
        }
        clear_grad();
        ifs.close();
    }
#undef RC


    vint gen(vint seed,int SL, bool rejecc_unk=0) {
        auto sig   = [](double x) -> double { return 1.0/(1.0 + exp(-x)); };

        int SZ = seed.size();
        vint seq(SL);
        random_device rd{};
        mt19937 gen(rd());

        vec x = vec::Zero(XS),
            z = vec::Zero(ZS),
            f = vec::Zero(HS),
            i = vec::Zero(HS),
        C_bar = vec::Zero(HS),
            O = vec::Zero(HS),
            v = vec::Zero(HS),
            y = vec::Zero(HS);

        // hidden state
        h.setZero();
        C.setZero();
        FOR(t,SZ) {
            x(seed[t]) = 1.0;
            z << h,x; // concatenate them
            f = (Wf.v * z + bf.v).unaryExpr(sig);
            i = ((Wi.v * z + bi.v).A).unaryExpr(sig);
            C_bar = tanh((WC.v * z + bC.v).A);
            C = f.A * C.A + i.A * C_bar.A;
            O = (Wo.v * z + bo.v).unaryExpr(sig);
            h = O.A * tanh(C.A);
            v = Wv.v * h + bv.v;
            y = exp(v.A)/exp(v.A).sum();

            x.setZero();
            if (t==SZ-1) {
                discrete_distribution<> prob_d(y.data(),y.data()+XS);
                seq[0] = prob_d(gen);
                x(seq[0]) = 1.0;
            }
        }

        for (int t=1;t<SL;t++) {
            z << h,x; // concatenate them
            f = (Wf.v * z + bf.v).unaryExpr(sig);
            i = ((Wi.v * z + bi.v).A).unaryExpr(sig);
            C_bar = tanh((WC.v * z + bC.v).A);
            C = f.A * C.A + i.A * C_bar.A;
            O = (Wo.v * z + bo.v).unaryExpr(sig);
            h = O.A * tanh(C.A);
            v = Wv.v * h + bv.v;
            y = exp(v.A)/exp(v.A).sum();

            x.setZero();
            discrete_distribution<> prob_d(y.data(),y.data()+XS);
            seq[t] = prob_d(gen);
            x(seq[t]) = 1.0;
        }
        return seq;
    }

    void bigLearn(const vint data,const int SL,const int epochs) {
        mat xs(XS,SL), // one hot input
            zs(ZS,SL), // actual input : one hot + hidden state ; C is
            fs(HS,SL),
            is(HS,SL),
        C_bars(HS,SL),
            Cs(HS,SL+1),
            Os(HS,SL),
            hs(HS,SL+1),
            vs(XS,SL),
            ys(XS,SL);
        vec dh_next(HS);
        vec dC_next(HS);
        int imax = data.size() - SL - 1;

        printf("will run %d batches over %d epochs\n",imax/SL,epochs);
        cout << smooth_loss << endl;

        int status_print = 0;
            random_device rd{};
            mt19937 gen(rd());
        auto sig   = [](double x) -> double { return 1.0/(1.0 + exp(-x)); };
        auto dsig  = [](double x) -> double { return x * (1.0-x); };
        auto dtanh = [](double x) -> double { return 1.0 - tanh(x)*tanh(x); };
        G_signal = 0;
        FOR(e,epochs) {
            h.setZero();
            C.setZero();
            for(int i=gen()%SL;i<imax;i+=SL) {
            if (G_signal) {
                cout << "\nprocess interupted"<<endl;
                if (safe_choice("would you like to stop? ")) return;
                else G_signal = 0;
            }
            // init
            hs.col(0) = h;
            Cs.col(0) = C;
            xs.setZero();
            double loss = 0.0;
            
            FOR(t,SL) { // forward pass
                xs(data[t+i],t) = 1.0;
                zs.col(t) << hs.col(t),xs.col(t);
                fs.col(t) = (Wf.v * zs.col(t) + bf.v).unaryExpr(sig);
                is.col(t) = (Wi.v * zs.col(t) + bi.v).unaryExpr(sig);
            C_bars.col(t) = tanh((WC.v * zs.col(t) + bC.v).A);
              Cs.col(t+1) = fs.col(t).A * Cs.col(t).A + is.col(t).A * C_bars.col(t).A;
                Os.col(t) = (Wo.v * zs.col(t) + bo.v).unaryExpr(sig);
              hs.col(t+1) = Os.col(t).A * tanh(Cs.col(t+1).A);
                vs.col(t) = Wv.v * hs.col(t+1) + bv.v;
                double Y = exp(vs.col(t).A).sum();
                ys.col(t) = exp(vs.col(t).A)/Y;
                    loss -= log(ys(data[t+i+1],t));
            }

            // prepare
            clear_grad();
            dh_next.setZero();
            dC_next.setZero();

            vec dv(XS),
                dh(HS),
                dO(HS),
                dC(XS),
            dC_bar(HS),
                di(HS),
                df(HS),
                dz(ZS);

            // backward pass
            ROF(t,SL) {
                dv = ys.col(t);
                dv(data[t+i+1]) -= 1.0;
                Wv.d += dv * hs.col(t+1).T;
                bv.d += dv;

                dh = Wv.v.T * dv + dh_next;
                dO = Os.col(t).unaryExpr(dsig).A * dh.A * tanh(Cs.col(t+1).A);
                Wo.d += dO * zs.col(t).T;
                bo.d += dO;

                dC = dC_next;
                dC.A += dh.A * Os.col(t).A * (tanh(Cs.col(t+1).A)).unaryExpr(dtanh);
                dC_bar = C_bars.col(t).unaryExpr(dtanh).A * dC.A * is.col(t).A;
                WC.d += dC_bar * zs.col(t).T;
                bC.d += dC_bar;

                di = dC.A * C_bars.col(t).A;
                di = is.col(t).unaryExpr(dsig).A * di.A;
                Wi.d += di * zs.col(t).T;
                bi.d += di;

                df = fs.col(t).unaryExpr(dsig).A * dC.A * Cs.col(t).A;
                Wf.d += df * zs.col(t).T;
                bf.d += df;

                dz = Wf.v.T * df + Wi.v.T * di + WC.v.T * dC_bar + Wo.v.T * dO;

                dh_next = dz.head(HS);
                dC_next = fs.col(t).A * dC.A;

            }

            clip_grad();
            h = hs.col(SL);
            C = Cs.col(SL);
            smooth_loss = smooth_loss*.999+loss*.001;

            // update parameters
            for (auto p:{&Wf,&bf,&Wi,&bi,&WC,&bC,&Wo,&bo,&Wv,&bv}) {
                p->m.A += p->d.A*p->d.A;
                p->v.A -= learning_rate * p->d.A / sqrt(p->m.A + 1e-8);
            }

            set_good((100.0-smooth_loss)/50.0); reset_line();
            printf("loss: %.2f - epoch %d progression %.2f",smooth_loss,e,i/(double)imax);
            fflush(stdout);
            if (status_print++>0) {
            // TODO
            // save state C H
            // reset_line + gen + endl
            // put back C H
                status_print = -20;
            }
        } }
        cout << "\ndone" << endl;
    }
};

#undef FOR
#undef ROF
#undef T
#undef A
#undef vint
#undef mat
#undef vec 

