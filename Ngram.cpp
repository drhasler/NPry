#pragma once
#include "Model.cpp"
#include <bits/stdc++.h>
using namespace std;

namespace ngram {
    // TODO make it a nested static member
    int nodes_cnt = 0;
}

class Ngram : public Model {
public:
    struct node {
        node* par;
        int dep,cnt;
        map<int,node*> nxt;
        node (node* p,int d) : par(p),dep(d) { cnt = 1; ngram::nodes_cnt++; }
        ~node() { for (auto x:nxt) delete x.second; ngram::nodes_cnt--; }
    } *root;

    Ngram(Piper* p) : Model(_Ngram, p) {
        ngram::nodes_cnt = 0;
        root = new node(nullptr,0);
        root->par = root;
    }

    node* percolate(node* cur,const vector<int>& seq) {
        for (int c:seq) {
            while (!cur->nxt.count(c)) // cannot go further
                cur = cur->par;
            cur = cur->nxt[c];
        }
        return cur;
    }
    // training input is entire file, no need output to
    void train(vector<int> seq, vector<int> outputs) {
        assert(outputs.empty());
        cout << " length of your snek: ";
        int n; cin >> n;
        // there might be a less brutal wae
        assert(seq.size()>n);
        // TODO beautify
        cout << "#" << flush;
        for (int c:seq) {
            if (!root->nxt.count(c)) root->nxt[c] = new node(root,1);
            root->nxt[c]->cnt++;
        }
        for (int x=1;x<n;x++) { // cur will have depth x, par dep = x-1 so that we add all x+1 nodes
            cout << "#" << flush;

            // percolate
            node* cur = root;
            for (int j=0;j<x;j++) 
                cur = cur->nxt[seq[j]];
            node* par = root;
            for (int j=1;j<x;j++) 
                par = par->nxt[seq[j]];

            // add next depth layer
            for (int j=x;j<seq.size();j++) {
                int c = seq[j];
                par = par->nxt[c]; // par temporarily has depth x
                if (!cur->nxt.count(c)) cur->nxt[c] = new node(par,x+1);
                cur->nxt[c]->cnt++;
                cur = par;
                par = cur->par;
            }
        }
        cout << "\nngram bereit" << endl;
    };

    void bigLearn(vector<int> seq,int x,int y) {
        cout << " how many grams do you want ? (N): " << flush;
        int n; cin >> n;
        assert(seq.size()>n);
        cout << "#" << flush;
        for (int c:seq) {
            if (!root->nxt.count(c)) root->nxt[c] = new node(root,1);
            root->nxt[c]->cnt++;
        }
        for (int x=1;x<n;x++) { // cur will have depth x, par dep = x-1 so that we add all x+1 nodes
            cout << "#" << flush;

            // percolate
            node* cur = root;
            for (int j=0;j<x;j++) 
                cur = cur->nxt[seq[j]];
            node* par = root;
            for (int j=1;j<x;j++) 
                par = par->nxt[seq[j]];

            // add next depth layer
            for (int j=x;j<seq.size();j++) {
                int c = seq[j];
                par = par->nxt[c]; // par temporarily has depth x
                if (!cur->nxt.count(c)) cur->nxt[c] = new node(par,x+1);
                cur->nxt[c]->cnt++;
                cur = par;
                par = cur->par;
            }
        }
        cout << "\nngram bereit" << endl;
    }

    // percolates and generate len # of tokens
    vector<int> gen(vector<int> seed ,int len, bool rejecc_unk=0) {
        node* cur = percolate(root,seed);
        cout << "input threshold value (~back edge probability)\n";
        int back; cin >> back;
        // make sure back is > 0
        random_device rd{};
        mt19937 gen{rd()};
        vector<int> seq(len);
        for (int i= 0; i<len; i++) { // len goto 0 :o
            vector<double> p;
            double tot = back + cur->cnt; //#(back_edge)+#(all children)
            p.push_back(back/tot);
            for (auto x:cur->nxt)
                p.push_back(x.second->cnt / tot);
            discrete_distribution<> prob_d(p.begin(),p.end());
            int y;
VOID_WORD:
            y = prob_d(gen);
            if (!y) { cur = cur->par; len++; } // truncate

            else { //we iterate over the children until we find the one choosen by the probability dist.
                for (auto x:cur->nxt) if (--y<=0) // == 0 but this prevents unexpected behavior
                {
                    if (rejecc_unk && x.first == 0){
                        goto VOID_WORD;
                    }
                    cur = x.second;
                    seq.push_back(x.first);
                    break;
                }
            }
        }
        return seq;
    }

#define RC(x) reinterpret_cast<char*>(&x),sizeof(x)
    void save(string fname) {
        // tokens serialization in model class
        ofstream ofs(fname,ios::binary);
        modelType Ty = mt;
        ofs.write(RC(Ty)); // assert ...
        ofs.write(RC(ngram::nodes_cnt));
        // BFS to make sure we have "parents" inserted
        int at = 0;
        // --------node,par_idx,tok
        queue<tuple<node*,int,int>> Q;
        Q.push({root,0,0});
        while (!Q.empty()) {
            auto [cur,par_idx,tok] = Q.front();
            Q.pop();
            ofs.write(RC(cur->cnt));
            ofs.write(RC(par_idx));
            ofs.write(RC(tok));
            for (auto x:cur->nxt) Q.push({x.second,at,x.first});
            at++;
        }
        ofs.close();
        assert(at==ngram::nodes_cnt);
    }
    void load (string fname) {
        ifstream ifs(fname,ios::binary);
        modelType Ty = mt;
        ifs.read(RC(Ty)); // assert ... 
        int nc; ifs.read(RC(nc));
        assert(ngram::nodes_cnt==1);// empty model
        int tok,cnt,par_idx;
        { // root is a special case
            ifs.read(RC(cnt));
            ifs.read(RC(par_idx));
            ifs.read(RC(tok));
            root->cnt = cnt;
        }
        vector<node*> vn;
        vn.push_back(root);
        for (int i=1;i<nc;i++) {
            ifs.read(RC(cnt));
            ifs.read(RC(par_idx));
            ifs.read(RC(tok));
            node* par = vn[par_idx];
            node* cur = new node(par,par->dep+1);
            par->nxt[tok] = cur;
            cur->cnt = cnt;
            vn.push_back(cur);
        }
        ifs.close();
    }
#undef RC
};
