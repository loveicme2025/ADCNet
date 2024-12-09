
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = r'/home/tym/dataset/LEVIR-CD/'  # Your Path
        elif data_name == 'BCDD':
            self.root_dir = r'/home/tym/dataset/BCDD/'
        elif data_name == 'levir':
            self.root_dir = r'/home/tym/dataset/levir-cd/'
        elif data_name == 'SYSU':
            self.root_dir = r'/home/tym/dataset/SYSU-CD/'    
        elif data_name == 'CDD':
            self.root_dir = r'/home/tym/dataset/CDD/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='CDD')
    # print(data.data_name)
    # print(data.root_dir)
    # print(data.label_transform)

