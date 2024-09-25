# 定义图的结构
class Graph:
    def __init__(self):
        self.num_subgraph = 3
        self.num_nodes = 4
        self.subgraph = {
            'head': {
                0: [0, 1, 2],
                1: [1, 3],
                2: []  # 子图3为空
            }
        }

# 定义稀疏矩阵类
class SparseMatrix:
    def __init__(self):
        self.value = []
        self.rowIndex = []
        self.colIndex = []
        self.rowNum = 0
        self.colNum = 0

# 构建子图稀疏矩阵
def subg_construct(graph):
    result = SparseMatrix()
    result.rowNum = graph.num_subgraph
    result.colNum = graph.num_nodes

    subgraph_id_span = []
    start = 0
    end = 0

    for i in range(graph.num_subgraph):
        list1 = graph.subgraph['head'][i]
        end = start + len(list1) - 1

        for j in range(len(list1)):
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(list1[j])

        if len(list1) > 0:
            subgraph_id_span.append((start, end))
        else:
            subgraph_id_span.append((graph.num_nodes, graph.num_nodes))

        start = end + 1

    return result

# 创建图对象
graph = Graph()

# 构建子图稀疏矩阵
sparse_matrix = subg_construct(graph)

# 输出稀疏矩阵的信息
print("Value:", sparse_matrix.value)
print("Row Index:", sparse_matrix.rowIndex)
print("Column Index:", sparse_matrix.colIndex)
print("Number of Rows:", sparse_matrix.rowNum)
print("Number of Columns:", sparse_matrix.colNum)
