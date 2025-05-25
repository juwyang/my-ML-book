// docs/js/mathjax-config.js
window.MathJax = {
  tex: {
    packages: {'[+]': ['boldsymbol', 'ams']}, // 添加 boldsymbol 和 ams 包
    inlineMath: [['\\(', '\\)']], // arithmatex generic 模式会使用这些
    displayMath: [['\\[', '\\]']]  // arithmatex generic 模式会使用这些
  },
  loader: {
    load: ['[tex]/boldsymbol', '[tex]/ams'] // 确保这些包被加载
  },
  startup: {
    // 这部分确保 MathJax 在 arithmatex 处理完之后再进行渲染
    // arithmatex 会将数学公式转换为 <script type="math/tex...">...</script>
    // MathJax 会查找这些 script 标签并渲染它们
    ready: () => {
      MathJax.startup.defaultReady();
      // 如果 arithmatex 使用了特定的 class (默认是 'arithmatex') 来包裹数学公式，
      // 你可能需要在这里触发对这些特定元素的 typeset，但通常 defaultReady 就够了。
      // 例如：MathJax.typesetPromise(document.querySelectorAll('.arithmatex'));
    }
  },
  svg: {
    fontCache: 'global' // 提高渲染速度
  }
};