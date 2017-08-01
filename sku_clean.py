import regex


class BaseReplacer:
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list

    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return text.strip()


class LowerCaseConverter(BaseReplacer):
    def transform(self, text):
        return str(text).lower()


class SpecMover(BaseReplacer):
    def __init__(self):
        spec_words_reg = '((([0-9.]+)\+)?([0-9.]+|[一二三四五六七八九十壹貳叁肆伍陸柒捌玖拾两双整])([mlkgMLKG千克毫升元套瓶盒支箱罐听件包袋片桶瓶个只单厅粒颗滴条抽枚块装]+))([*xX×][0-9.]+|[/])?([mlkgMLKG千克毫升元套瓶盒支箱罐听件包袋单片桶瓶个只厅粒颗滴条抽枚块装]+)?'

        self.pattern_replace_pair_list = [(spec_words_reg, r'')]


class TrashWords(BaseReplacer):
    def __init__(self):
        trash_words_reg = ['\s', '(满)?[0-9.]+[立再减-]+[0-9.]+(元)?', '[0-9.]+件[0-9.]+折','国产', '优惠装', '授权', '行货', '原装', '海淘',
                           '(荷兰(?!朵))', '正品', '包邮', '休闲零食品', '办公室',
                           '爆款', '专柜', '直邮', '原装', '官方标[配]?',
                          '均码', '正常规格', '任选', '天猫超市', '苏宁易购', '公司出品', '.包装', '领券',
                           '雅诗兰黛', 'estee lauder', 'esteelauder','\\',
                           '阿迪达斯', 'adidas', '阿迪', '超市', '[杯罐桶]装', '易碎品', '乐事',
                           'kiehl(\s)?(s)?', '科颜氏', '契尔氏', '苏宁易购超市', '马来西亚', '良品', '铺子', '-']

        self.pattern_replace_pair_list = [('|'.join(trash_words_reg), r'')]


class WordReplacer(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [(r'の', r'之'), (r'貳', r'二'),
                                          (r'叁', r'三'), (r'肆', r'四'),
                                          (r'伍', r'五'), (r'陸', r'六'),
                                          (r'柒', r'七'), (r'捌', r'八'),
                                          (r'玖', r'九'), (r'拾', r'十'),
                                          (r'kielhs', r'kiehls'), (r'é', 'e')]


class SymbolsCleaner(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = \
            [(r'[·＇+＋‘’!@#$%&＆/*?=_－—<>.··．﹒．:：;′()（）【】，。！？,“”●]', r"")]
