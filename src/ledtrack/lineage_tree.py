# # ETE is distributed under the GPL copyleft license (2008-2015).
# #
# # If you make use of ETE in published work, please cite:
# #
# # Jaime Huerta-Cepas, Joaquin Dopazo and Toni Gabaldon.
# # ETE: a python Environment for Tree Exploration. Jaime BMC
# # Bioinformatics 2010,:24doi:10.1186/1471-2105-11-24

# from ete3 import Tree, NodeStyle, TreeStyle
import numpy as np

from ete3.treeview.qt4_render import _TreeScene, render, init_tree_style


class LineageTree():
    def __init__(self, track, track_global_index=None, node_size=20, if_scene=True):
        self.track = track
        self.track_global_index = np.linspace(0, track.shape[-1]-1, track.shape[-1], dtype=np.uint16) \
            if track_global_index is None else track_global_index
        self.node_size = node_size

        self.nstyle = NodeStyle()
        self.nstyle["size"] = 0
        self.nstyle["fgcolor"] = 'white'

        self.nstyle0 = NodeStyle()
        self.nstyle0["size"] = 1
        self.nstyle0["fgcolor"] = 'lightgray'
        self.nstyle0["hz_line_type"] = 1
        self.nstyle0["hz_line_color"] = "#cccccc"
        self.nstyle0["bgcolor"] = "Khaki"

        self.nstyle1 = NodeStyle()
        self.nstyle1["shape"] = "sphere"
        self.nstyle1["size"] = self.node_size
        self.nstyle1["fgcolor"] = "lightskyblue"
        self.nstyle1["hz_line_type"] = 0
        self.nstyle1["hz_line_color"] = "#FFFFFFF"

        self.nstyle2 = NodeStyle()
        self.nstyle2["shape"] = "sphere"
        self.nstyle2["size"] = self.node_size
        self.nstyle2["fgcolor"] = "lightcoral"
        self.nstyle2["hz_line_type"] = 0
        self.nstyle2["hz_line_color"] = "#FFFFFFF"

        self.get_lineage_tree(self.track, track_global_index=self.track_global_index)

        if if_scene:
            self.ts = TreeStyle()
            self.ts.mode = "c" ##"c" or "r"
            self.ts.arc_start = -180  # 0 degrees = 3 o'clock
            self.ts.arc_span = 360 if track.shape[1] > 90 else track.shape[1] * 4
            # self.ts.scale = 120  # 120 pixels per branch length unit
            # self.ts.branch_vertical_margin = 10  # 10 pixels between adjacent branches
            # # self.ts.rotation = 90
            self.ts.show_leaf_name = False
            # self.ts.show_branch_length = True
            # self.ts.show_branch_support = True
            self.get_scene(self.tree, layout=layout, tree_style=self.ts)

    def init_scene(self, t, layout, ts):
        ts = init_tree_style(t, ts)
        if layout:
            ts.layout_fn = layout
        scene = _TreeScene()
        # ts._scale = None
        return scene, ts

    def get_scene(self, t, layout=None, tree_style=None):
        """ Interactively shows a tree."""
        self.scene, img = self.init_scene(t, layout, tree_style)
        tree_item, n2i, n2f = render(t, img)
        self.scene.init_values(t, img, n2i, n2f)

        tree_item.setParentItem(self.scene.master_item)
        self.scene.addItem(self.scene.master_item)


    def get_lineage_tree(self, track, track_global_index):
        """
        根据细胞追踪数据构建谱系树。

        从轨迹数据中提取细胞分裂关系，构建树形结构来表示细胞谱系。
        支持根节点创建、初始细胞添加、新细胞检测以及细胞分裂路径的追踪。

        Args:
            track: 二维数组，shape 为 (帧数, 细胞数)，存储每帧的细胞ID标识。
                   -1 表示无母细胞的细胞（可能是新生成的细胞）
            track_global_index: 全局索引数组

        Returns:
            无返回值，结果存储在 self.tree 属性中。
        """
        name = -1
        self.tree = Tree(name='root', dist=0)
        self.tree.track = ['#root#']
        self.tree.frame = -1
        # self.tree.end_leaf = True
        # self.tree.support = 0
        self.tree.trackid = -1
        self.tree.n = -1
        self.tree.set_style(self.nstyle)
        name += 1

        cells = np.unique(track[0])
        cells = cells[cells != -1]
        for cell in cells:
            child = self.tree.add_child(name=name, dist=1)
            child.frame = 0
            child.end_leaf = False
            child.set_style(self.nstyle1)
            child.n = cell
            # child.support = cell
            name += 1

        uniq_c = {x: [] for x in range(track.shape[0])}
        for new_cell in track.T[track[0] == -1]:
            ID = np.where(new_cell != -1)[0]
            if new_cell[min(ID)] in uniq_c[min(ID)]:
                continue
            else:
                uniq_c[min(ID)].append(new_cell[min(ID)])
            mother = self.tree.add_child(name=-1, dist=min(ID))
            mother.n = new_cell[min(ID)]
            mother.frame = 0
            mother.end_leaf = True
            # mother.support = new_cell[min(ID)]
            mother.trackid = track_global_index[np.where(track[int(mother.frame + mother.dist)] == mother.n)[0]]
            # mother.track = [f'#id: {mother.trackid}#frame: '+str(mother.frame)+'-'+
            #                 str(int(mother.frame + mother.dist - 1)) + '#'] + \
            #                 new_cell[mother.frame: int(mother.frame + mother.dist)].tolist() + ['#']
            # mother.track = new_cell[mother.frame: int(mother.frame + mother.dist)].tolist()
            mother.track = [-1]*int(mother.dist)
            child = mother.add_child(name=name, dist=1)
            name += 1
            child.n = new_cell[min(ID)]
            child.frame = min(ID)
            child.trackid = mother.trackid
            child.end_leaf = False
            # child.track = [f'#id: {child.trackid}#frame: ' + str(child.frame) + '-' +
            #                 str(int(child.frame + child.dist - 1)) + '#'] + \
            #                new_cell[child.frame: int(child.frame+child.dist)].tolist() + ['#']
            child.track = new_cell[child.frame: int(child.frame + child.dist)].tolist()

            # child.support = new_cell[min(ID)]
            # Applies the same static style to all nodes in the tree. Note that,
            # if "nstyle" is modified, changes will affect to all nodes
            mother.set_style(self.nstyle0)
            child.set_style(self.nstyle1)

        for leaf in self.tree.iter_leaves():
            if leaf.end_leaf:
                continue
            mother_cell, frame = leaf.n, leaf.frame
            cell_line = track[frame:, track[frame] == mother_cell]
            if len(cell_line) == 0:
                # leaf.track = []
                leaf.end_leaf = True
                continue
            unique_cell = [np.unique(cell_l) for cell_l in cell_line]

            branch = np.array([len(cs) for cs in unique_cell])

            assert np.all(branch[1:] - 2 * branch[:-1] < 1), 'Branching ERROR!!!'

            mask = np.logical_and(branch == 1, cell_line[:, 0] != -1)
            #  因为ind->len所以要 + 1
            leaf.dist = np.where(mask == 1)[0].max() + 1 if np.any(mask) else 1
            # leaf.support = unique_cell[int(leaf.dist) - 1][0]
            leaf.trackid = track_global_index[np.where(track[leaf.frame] == leaf.n)[0]]
            # leaf.track = [f'#id: {leaf.trackid}#frame:' + str(leaf.frame) + '-' +
            #               str(int(leaf.frame + leaf.dist - 1)) + '#'] + [mother_cell] + \
            #              np.array(cell_line[:int(leaf.dist), 0]).reshape(-1).tolist() + ['#']
            leaf.track = np.array(cell_line[:int(leaf.dist), 0]).reshape(-1).tolist()
            leaf.end_leaf = True

            idxs = np.where(branch == 2)[0]
            if len(idxs) > 0:
                leaf.set_style(self.nstyle2)
                for cell in unique_cell[idxs[0]]:
                    col = np.where(cell_line[branch == 2][0] == cell)[0]
                    child = leaf.add_child(name=name, dist=1)
                    child.n = cell
                    name += 1
                    child.frame = leaf.frame + idxs[0]
                    # child.frame = int(leaf.frame + leaf.dist)
                    child.end_leaf = False
                    # leaf.support = cell
                    # child.track = [f'#id: {leaf.trackid[col]}#frame: ' + str(child.frame) + '-' +
                    #                str(int(child.frame + child.dist - 1)) + '#'] + \
                    #               cell_line[branch == 2][:int(child.dist), col[0]].tolist() + ['#']
                    child.track = cell_line[branch == 2][:int(child.dist), col[0]].tolist()
                    child.set_style(self.nstyle1)


from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle

def layout(node):
    if node.is_leaf():
        N = AttrFace("name", fsize=30)
        faces.add_face_to_node(N, node, 0, position="aligned")

def get_example_tree():

    # Set dashed blue lines in all leaves
    nst1 = NodeStyle()
    nst1["bgcolor"] = "LightSteelBlue"
    nst2 = NodeStyle()
    nst2["bgcolor"] = "Moccasin"
    nst3 = NodeStyle()
    nst3["bgcolor"] = "DarkSeaGreen"
    nst4 = NodeStyle()
    nst4["bgcolor"] = "Khaki"


    t = Tree("((((a1,a2),a3), ((b1,b2),(b3,b4))), ((c1,c2),c3));")
    for n in t.traverse():
        n.dist = 0

    n1 = t.get_common_ancestor("a1", "a2", "a3")
    n1.set_style(nst1)
    # n2 = t.get_common_ancestor("b1", "b2", "b3", "b4")
    # n2.set_style(nst2)
    # n3 = t.get_common_ancestor("c1", "c2", "c3")
    # n3.set_style(nst3)
    n4 = t.get_common_ancestor("b3", "b4")
    n4.set_style(nst4)
    ts = TreeStyle()
    ts.layout_fn = layout
    ts.show_leaf_name = False

    ts.mode = "c"
    ts.root_opening_factor = 1
    return t, ts

if __name__ == "__main__":
    t, ts = get_example_tree()
    #t.render("node_background.png", w=400, tree_style=ts)
    t.show(tree_style=ts)