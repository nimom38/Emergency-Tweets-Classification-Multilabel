#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef pair<int,int> pii;
typedef pair<ll, ll> pll;
typedef vector<pii> vii;
typedef vector<pll> vll;

#define PB push_back
#define F first
#define S second
#define MP make_pair
#define endl '\n'

const int inf = 2000000000;
const ll infLL = 9000000000000000000;
#define MOD 1000000007

#define optimize() ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);

//
//debug
template<typename F,typename S>ostream&operator<<(ostream&os,const pair<F,S>&p){return os<<"("<<p.first<<", "<<p.second<<")";}
template<typename T>ostream&operator<<(ostream&os,const vector<T>&v){os<<"{";for(auto it=v.begin();it!=v.end();++it){if(it!=v.begin())os<<", ";os<<*it;}return os<<"}";}
template<typename T>ostream&operator<<(ostream&os,const set<T>&v){os<<"[";for(auto it=v.begin();it!=v.end();++it){if(it!=v.begin())os<<",";os<<*it;}return os<<"]";}
template<typename T>ostream&operator<<(ostream&os,const multiset<T>&v) {os<<"[";for(auto it=v.begin();it!=v.end();++it){if(it!=v.begin())os<<", ";os<<*it;}return os<<"]";}
template<typename F,typename S>ostream&operator<<(ostream&os,const map<F,S>&v){os<<"[";for(auto it=v.begin();it!=v.end();++it){if(it!=v.begin())os<<", ";os<<it->first<<" = "<<it->second;}return os<<"]";}
#define dbg(args...) do {cerr << #args << " : "; faltu(args); } while(0)
void faltu(){cerr << endl;}
template<typename T>void faltu(T a[],int n){for(int i=0;i<n;++i)cerr<<a[i]<<' ';cerr<<endl;}
template<typename T,typename...hello>void faltu(T arg,const hello&...rest){cerr<<arg<<' ';faltu(rest...);}
//#else
//#define dbg(args...)

void func1()
{
	string s;
	cin >> s;
	s.erase( s.begin() );
	s.erase( s.begin() );
	s.pop_back();
	s.pop_back();
	cout << s << '\t';
}

void func2()
{
	string s;
	cin >> s;
	s.erase( s.begin() );
	s.pop_back();
	s.pop_back();
	cout << s << '\t';
}

void func3()
{
	string s;
	cin >> s;
	s.pop_back();
	cout << s << '\t';
	cin >> s;
	s.pop_back();
	cout << s << '\t';
	cin >> s;
	s.pop_back();
	cout << 0.5 << '\t';
}

void func4()
{
	while(1) {
		string s;
		cin >> s;
		if( s == "\'myrun\']" ) {
			cout << '\t';
			s.erase( s.begin() );
			s.pop_back();
			s.pop_back();
			cout << s << endl;
			return;
		}
		for( int i = 0; i < (int)s.size(); ++i ) {
			if( s[i] == '\'' ) {
				s.erase( s.begin()+i );
				--i;
			}
			else if( s[i] == ' ' ) {
				s.erase( s.begin()+i );
				--i;
			}
		}
		if( s[(int)s.size()-2] == ']' && s.back() == ',' ) s.pop_back();
		cout << s;
	}
}

int main()
{
	optimize();
	freopen( "input.txt", "r", stdin );
	freopen( "output.txt", "w", stdout );
	for( int i = 0; i < 5543; ++i ) {
		func1();
		func2();
		func3();
		func4();
	}
}






















