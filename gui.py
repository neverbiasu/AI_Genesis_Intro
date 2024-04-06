from tkinter import *
from tkinter import ttk
# 创建主窗口
root = Tk()
root.title('321——tools')

# 设置窗口的大小和位置
root.geometry('800x600')  # 你可以根据实际UI设计调整大小

mainframe = ttk.Frame(root, padding='3 2 12 12')
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# 添加标签和文本框
# Create subframes for the left and right sections
left_frame = ttk.Frame(mainframe, padding="3 3 12 12", relief="sunken")
right_frame = ttk.Frame(mainframe, padding="3 3 12 12", relief="sunken")

left_frame.grid(column=0, row=0, sticky=(N, W, E, S))
right_frame.grid(column=1, row=0, sticky=(N, W, E, S))

label_bottom_left = ttk.Label(root, text='大人取用上缴物')
text_bottom_left = ttk.Entry(root)

label_bottom_right = ttk.Label(root, text='生活区拾到物品上缴处')
text_bottom_right = ttk.Entry(root)

# 使用grid布局管理器来布局标签和文本框
# label_top_left.grid(row=0, column=0, padx=10, pady=10)
# entry_top_left.grid(row=1, column=0, padx=10, pady=10)

# label_top_right.grid(row=0, column=1, padx=10, pady=10)
# entry_top_right.grid(row=1, column=1, padx=10, pady=10)

label_bottom_left.grid(row=2, column=0, padx=10, pady=10)
text_bottom_left.grid(row=3, column=0, padx=10, pady=10)

label_bottom_right.grid(row=2, column=1, padx=10, pady=10)
text_bottom_right.grid(row=3, column=1, padx=10, pady=10)

# 运行主循环
root.mainloop()