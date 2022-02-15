#include <iostream>
#include <vector>
using namespace std;
/*
src在dst的分布,比如src[0][0]在且仅在dst[0:4][0:4]
 * */
const int ORI=8;     //8个方向
const int SEG=2;     //2个分割
const int INDEX=16;  //16索引

// const int ORI=16;     //8个方向
// const int SEG=4;     //2个分割
// const int INDEX=16;  //16索引
struct Node 
{
    int value;
    int prev;
    int next;
    void print()
    {
        cout<<prev<<","<<value<<","<<next<<endl;
    }
};
int main()
{
    std::vector<Node> nodes(ORI);
    for(int i=0; i<ORI; i++)
    {
        nodes[i].value = (1 << i);
        nodes[i].prev = i-1;
        nodes[i].next = i+1;
    }
    nodes[0].prev = ORI-1;
    nodes[ORI-1].next = 0;

    /***
     * 打印所有节点
     * 7,1,1
       0,2,2
       1,4,3
       2,8,4
       3,16,5
       4,32,6
       5,64,7
       6,128,0
     * */
    for(auto node:nodes)
        node.print();

    uint8_t LUT[ORI*SEG*INDEX] = {0};
    uint8_t LUT_RIGHT[] = {0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                           0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                           0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                           0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
                           0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4,
                           0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4,
                           0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4,
                           0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};

    for(int i=0; i<ORI; i++)            // 8 ori
        for(int m=0; m<SEG; m++)        // 2 seg
            for(int n=0; n<INDEX; n++)  // 16 index
            { 
                int index=n+m*INDEX+i*INDEX*SEG;
                if(n==0)  // no ori
                { 
                   LUT[index] = 0;
                   continue;
                }

                int res = (n << (m*4));  //4 here comes from 2^4=16 (INDEX)
/*
m	0															
n	0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15
res		1	2	3	4	5	6	7	8	9	10	11	12	13	14	15
LUT	0	4	3	4	2	4	3   4   1   4    3  4   2   4   3   4									
m	1															
n	0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15
res		16	32	48	64	80	96	112	128	144	160	176	192	208	224	240
LUT	0	0   1   1   2   2   2   2   3   3   3   3   3   3   3   3
*/														
                auto current_node_go_forward = nodes[i];
                auto current_node_go_back = nodes[i];
                int angle_diff = 0;
                while(1)
                {
                    if((current_node_go_forward.value & res) > 0 ||(current_node_go_back.value & res) > 0)
                    {
                        break;
                    }
                    else
                    {
                        current_node_go_back = nodes[current_node_go_back.prev];
                        current_node_go_forward = nodes[current_node_go_forward.next];
                        angle_diff ++;
                    }
                }
                LUT[index] = 4 - angle_diff;
                if(i==0&&m==0)
                {
                    cout<<res<<","<<int(LUT[index])<<endl; 
                };
            }
    for(int i=0; i<ORI; i++)
    {
        for(int m=0; m<SEG*INDEX; m++)
            cout << int(LUT[i*SEG*INDEX + m]) << ", ";
        cout << "\n";
    }
    return 0;
}

