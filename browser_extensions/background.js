

function extractURLsFromHtml(rawHTML){

  var anchors = /<a\s[^>]*?href=(["']?)([^\s]+?)\1[^>]*?>/ig;
  var links = [];
  rawHTML.replace(anchors, function (_anchor, _quote, url) {
    links.push(url);
  });
  
  console.log(links);
  return links
  }

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("mesage recived to background.js  = "+ message);
    //

});



// on webpage changed

chrome.tabs.onUpdated.addListener(function
  (tabId, changeInfo, tab) {
    //  do something with it (like read the url)
    if (changeInfo.status) {
      console.log("The URL has changed");
      console.log("status = "+changeInfo.status)
    }
    if(changeInfo.status == 'complete'){
      console.log('true')
      //extractButton(null)

      setTimeout( ()=>
        chrome.tabs.sendMessage( tabId, {
          msg: 'urlchange',
          status: changeInfo.status
        })
      ,1000);

    }
  }
);


chrome.tabs.query({ active: true, currentWindow: true }, function(tabs){
  console.log(tabs)
  tabs.forEach(element => { console.log(element)})
});







