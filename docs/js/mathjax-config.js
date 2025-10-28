window.MathJax = {
  tex: {
    packages: {'[+]': ['boldsymbol', 'ams', 'amssymb']}, // 添加必要的包
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    macros: {
      // 为常用命令添加宏定义，确保兼容性
      bm: ['{\\boldsymbol{#1}}', 1],
      argmax: '\\operatorname{argmax}',
      argmin: '\\operatorname{argmin}',
      vec: ['{\\boldsymbol{#1}}', 1]
    },
    processEscapes: true, // 处理双反斜杠换行
    multlineWidth: '85%'
  },
  loader: {
    load: ['[tex]/boldsymbol', '[tex]/ams', '[tex]/amssymb'] // 确保包被加载
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      // 手动触发 arithmatex 元素的渲染
      MathJax.typesetPromise(document.querySelectorAll('.arithmatex')).catch(err => console.log(err));
    }
  },
  svg: {
    fontCache: 'global',
    scale: 1.1 // 可选：调整字体大小以获得更好的显示效果
  }
};