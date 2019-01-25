#pragma once
#include <bits/stdc++.h>
#include "../utils/IO.cpp"
#define FOR(a,b) for(int a=0;a<(int)(b);a++)
using namespace std;
//#define PIPER_TEST

enum tokenType {STR,CHAR};

class Piper {
public:
    const string hardPunk = "?!.\n";
    const string slurp = ",.?!:;\n";
    string repr;
    int unique_wc = 0;
    int tot_wc = 0;
    const tokenType tT;
    Piper(tokenType x) : tT(x) {}
    virtual vector<int> tokenize(const string& txt) = 0;
    virtual string untokenize(const vector<int>& seq) = 0;
};

class CPiper : public Piper {
public:
    const char default_w = '_';
    map<char,int> tok;
    map<int,char> untok;

    CPiper();
    void add_occurences(string fname);
    vector<int> tokenize(const string&);
    string untokenize(const vector<int>&);
};

class SPiper : public Piper {
public:
    const string default_w = "snek";
    map<string,int> tok;
    map<int,string> untok;

    SPiper();
    void add_occurences(string fname);
    vector<int> tokenize(const string&);
    string untokenize(const vector<int>&);
};

void CPiper::add_occurences(string fname) {
    string wale = read_text(fname);
    for (char c:wale) { if (!tok[c]++) unique_wc++; tot_wc++; }
}

void SPiper::add_occurences(string fname) {
    string line; // cur line
    string w; // curent word
    int state = 2; // things we dont care about
    auto reset = [&](char c){ w = string(1,c); state = isalpha(c)?0:ispunct(c)?1:2; };
    auto add = [&](){ if (!tok[w]++) unique_wc++; tot_wc++; };
    ifstream ifs(fname);

    while (getline(ifs,line)) {
        for (char c:line) {
            c = tolower(c);
            switch (state) {
                case 0: // abc sequence
                    if (isalpha(c)) w += c; // extend
                    else { add(); reset(c); }
                    break;
                case 1:  // punctuation
                    add(); reset(c);
                    break;
                default:
                    reset(c); // numbers and weird chars
            }
        }
        // dont forget about the last word
        if (state==0 || state==2) add();
        w = "\n"; add();
        w.clear(); state = 2;
    }
}

CPiper::CPiper() : Piper(CHAR) {
    unique_wc = 0;
    cout << "char piper: building voc, please input filename:\n"<<flush;
    string fname; cin >> fname;
    while (fname!="ok" || unique_wc==0) {
        if (exists(fname)) {
            add_occurences(fname);
            printf("currently %d unique tokens, %d seen in total\n",unique_wc,tot_wc);
        } else {
            cout << "bad filename, at cur folder:\n";
            system("ls > pout/last_cmd.out");
            string lsout = read_text("pout/last_cmd.out");
            cout << lsout << flush;
        }
        cout << "input new file or ok: "<<flush;
        cin >> fname;
    }
    vector<pair<char,int>> temp;
    // we first convert map to vector
    transform(tok.begin(),tok.end(),back_inserter(temp),
            [](const auto& x){return x;});
    // nth element is faster if we are sure that we split only once
    sort(temp.begin(),temp.end(),
            [](const auto& a,const auto& b){return a.second > b.second;});
    partial_sum(temp.begin(),temp.end(),temp.begin(),
            [](const auto& a,const auto& b)->pair<char,int>
            {return {b.first,a.second+b.second};});
    // now tok will really map a char to its token
    tok.clear();
    // we counted the right number of tokens
    assert(unique_wc==temp.size());
    assert(tot_wc==temp[unique_wc-1].second);
    int tokeep;
    do {
        tokeep = safe_int("voc size",1,unique_wc);
        printf("with voc_size %d, you will have %.2f%% tokenized\n",
                tokeep, (double)temp[tokeep-1].second * 100.0 / (double)tot_wc);
    } while (!safe_choice("happy with it? "));
    for (int i=0;i<tokeep;i++) {
        tok[temp[i].first] = i+1;
        untok[i+1] = temp[i].first;
    }
    // default token
    unique_wc = tokeep+1;
    untok[0] = default_w;
    // string representation
    repr = "<char,"+to_string(unique_wc)+">";
}

SPiper::SPiper() : Piper(STR) {
    unique_wc = 0;
    cout << "building voc, please input filename: "<<flush;
    string fname; cin >> fname;
    while (fname!="ok" || unique_wc==0) {
        if (exists(fname)) {
            add_occurences(fname);
            printf("currently %d unique tokens, %d seen in total\n",unique_wc,tot_wc);
        } else {
            cout << "bad filename, at cur folder:\n";
            system("ls > pout/last_cmd.out");
            string lsout = read_text("pout/last_cmd.out");
            cout << lsout << flush;
        }
        if (unique_wc) cout << "input new file or ok: "<<flush;
        else cout << "pls input correct filename: " << flush;
        cin >> fname;
    }
    vector<pair<string,int>> temp;
    // we first convert map to vector
    transform(tok.begin(),tok.end(),back_inserter(temp),
            [](const auto& x){return x;});
    // nth element is faster if we are sure that we split only once
    sort(temp.begin(),temp.end(),
            [](const auto& a,const auto& b){return a.second > b.second;});
    partial_sum(temp.begin(),temp.end(),temp.begin(),
            [](const auto& a,const auto& b)->pair<string,int>
            {return {b.first,a.second+b.second};});
    // we counted the right number of tokens
    assert(unique_wc==temp.size());
    assert(tot_wc==temp[unique_wc-1].second);
    // now tok will really map a string to its token
    tok.clear();
    int tokeep;
    do {
        tokeep = safe_int("voc size",1,unique_wc);
        printf("with voc_size %d, you will have %.2f%% tokenized\n",
                tokeep, (double)temp[tokeep-1].second * 100.0 / (double)tot_wc);
    } while (!safe_choice("happy with it? "));
    for (int i=0;i<tokeep;i++) {
        tok[temp[i].first] = i+1;
        untok[i+1] = temp[i].first;
    }
    // default token
    unique_wc = tokeep+1;
    untok[0] = default_w;
    // string representation
    repr = "<string,"+to_string(unique_wc)+">";
}

vector<int> CPiper::tokenize(const string& txt) {
    int SL = txt.size();
    vector<int> ans(SL);
    FOR(i,SL) ans[i] = tok.count(txt[i]) ? tok.at(txt[i]) : 0;
    return ans;
}

vector<int> SPiper::tokenize(const string& txt) {
    vector<int> ans;
    istringstream iss(txt);
    string line;
    string w; // curent word
    int state = 2; // things we dont care about
    auto reset = [&](char c){ w = string(1,c); state = isalpha(c)?0:ispunct(c)?1:2; };
    auto add = [&](){ ans.push_back(tok.count(w) ? tok.at(w) : 0); };

    while (getline(iss,line)) {
        for (char c:line) {
            c = tolower(c);
            switch (state) {
                case 0: // abc sequence
                    if (isalpha(c)) w += c; // extend
                    else { add(); reset(c); }
                    break;
                case 1:  // punctuation
                    add(); reset(c);
                    break;
                default:
                    reset(c); // numbers and weird chars
            }
        }
        // dont forget about the last word
        if (state==0 || state==2) tok[w]++;
        w.clear(); state = 2;
        ans.push_back(tok["\n"]);
    }
    return ans;
}

string CPiper::untokenize(const vector<int>& seq) {
    string ans;
    // TODO remove first space at each line begining
    bool needUp = 1;
    for (int i:seq) {
        char c = untok[i];
        ans += needUp ? toupper(c) : c;
        if (isalpha(c)) needUp = 0;
        needUp |= (hardPunk.find(c) != string::npos);
    }
    return ans;
}

string SPiper::untokenize(const vector<int>& seq) {
    string ans;
    // TODO remove first space at each line begining
    bool needUp = 1;
    for (int i:seq) {
        string s = untok[i];
        char c = s[0];
        if (slurp.find(c)==string::npos) ans+=' ';
        if (needUp) s[0] = toupper(c);
        ans += s;
        if (isalpha(c)) needUp = 0;
        needUp |= (hardPunk.find(c) != string::npos);
    }
    return ans;
}

#ifdef PIPER_TEST
int main() {
    Piper* perry = new CPiper();
    string kek;
    cin.ignore();
    cout << "donnes moi du kek!\n";
    getline(cin,kek);
    vector<int> snek = perry -> tokenize(kek);
    cout << perry->untokenize(snek);
}
#endif
