"""Registered spatial components of Chapter 2 method A."""

ARMS = {
    'A_FULL': dict(method='A', variant='A_FULL', temporal='fixed',
                   purpose='完整方法，固定各期总量并学习对象配置'),
    'A_RANDOM_MATCH': dict(method='A', variant='A_RANDOM_MATCH', temporal='fixed',
                           purpose='用同维随机特征替换多尺度响应表征'),
    'A_SOFTMAX': dict(method='A', variant='A_SOFTMAX', temporal='fixed',
                      purpose='用有上限的比例配置替换投影配置'),
    'A_DIRECT': dict(method='A', variant='A_DIRECT', temporal='fixed',
                     purpose='直接输出对象配置分数，替换响应特征结构'),
}


def arm(name):
    try:
        return dict(ARMS[name])
    except KeyError as error:
        raise ValueError('Unknown chapter-2 ablation: '+str(name)) from error
