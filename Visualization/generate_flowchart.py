import graphviz

def create_flowchart():
    # Khởi tạo đồ thị
    dot = graphviz.Digraph(comment='Experiment Flowchart', format='png')
    
    # Cấu hình chung
    dot.attr(rankdir='TD', size='10,14')
    dot.attr('node', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='11')
    
    # Định nghĩa các nút (Nodes)
    dot.node('Start', 'Bắt đầu', shape='oval')
    dot.node('Init', 'Nhập Dataset D và các tập tham số:\nN, S, G, Φ_G, M', shape='parallelogram')
    dot.node('Step1', 'Bước 1: Chia Dataset D thành 10 folds\n(Đảm bảo các mẫu cùng ID nằm trong cùng fold)', shape='box')

    # Các vòng lặp tô màu hồng nhạt
    loop_style = {'shape': 'diamond', 'style': 'filled', 'fillcolor': '#f9f2f4', 'color': '#d0a0b0', 'penwidth': '2'}
    
    dot.node('LoopK', 'k = 1..10?', **loop_style)
    dot.node('K_Init', 'Khởi tạo D_test = Fold(k)\nvà D_train_pool = D \\ Fold(k)', shape='box')

    dot.node('LoopN', 'N ∈ N?', **loop_style)
    dot.node('LoopR', 'r = 1..10?', **loop_style)
    
    dot.node('Sample', 'Bước 2.5: Lấy ngẫu nhiên N mẫu/class\ntừ D_train_pool → D_seed', shape='box')

    dot.node('LoopConfig', 'Duyệt tổ hợp cấu hình (G, φ, S)?', **loop_style)
    dot.node('Generate', 'Bước 3: Huấn luyện G(φ) trên D_seed.\nSinh dữ liệu tổng hợp → D_synth', shape='box')
    
    dot.node('LoopM', 'M ∈ M?', **loop_style)

    dot.node('GridSearch', 'Bước 4: Grid Search (5-fold Cross-Validation)\ntrên D_synth → Lấy theta_opt', shape='box')
    dot.node('TrainM', 'Train M using theta_opt', shape='box')
    dot.node('Evaluate', 'Evaluate on D_test\n→ Save Score(k, N, r, G, S, φ, M)', shape='box')

    dot.node('Step5', 'Bước 5: Tổng hợp thống kê\nTính trung bình 100 lần (10 folds x 10 seeds)\n→ Ma trận P', shape='box')
    dot.node('Output', 'Trả về Ma trận P', shape='parallelogram')
    dot.node('End', 'Kết thúc', shape='oval')

    # Định nghĩa các luồng (Edges)
    dot.edge('Start', 'Init')
    dot.edge('Init', 'Step1')
    
    dot.edge('Step1', 'LoopK')
    dot.edge('LoopK', 'K_Init', label='k nhánh')
    
    dot.edge('K_Init', 'LoopN')
    dot.edge('LoopN', 'LoopR', label='N nhánh')
    dot.edge('LoopR', 'Sample', label='r nhánh')
    
    dot.edge('Sample', 'LoopConfig')
    dot.edge('LoopConfig', 'Generate', label='Từng tổ hợp cấu hình')
    
    dot.edge('Generate', 'LoopM')
    dot.edge('LoopM', 'GridSearch', label='M nhánh')
    
    dot.edge('GridSearch', 'TrainM')
    dot.edge('TrainM', 'Evaluate')

    # Các đường lặp quay ngược (Back-edges)
    dot.edge('Evaluate', 'LoopM')
    dot.edge('LoopM', 'LoopConfig', label='Hết M')
    dot.edge('LoopConfig', 'LoopR', label='Hết cấu hình')
    dot.edge('LoopR', 'LoopN', label='Hết r')
    dot.edge('LoopN', 'LoopK', label='Hết N')

    # Thoát vòng lặp
    dot.edge('LoopK', 'Step5', label='Hết k')
    dot.edge('Step5', 'Output')
    dot.edge('Output', 'End')

    # Lưu và hiển thị (tạo ra file flowchart.png và flowchart.gv)
    dot.render('experiment_flowchart', view=False, cleanup=False)
    print("Flowchart saved to experiment_flowchart.png")

if __name__ == '__main__':
    create_flowchart()
