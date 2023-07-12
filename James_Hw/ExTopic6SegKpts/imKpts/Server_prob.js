const https = require('https');

https.get('https://api.ipify.org', (res) => {
  let data = '';

  // A chunk of data has been received.
  res.on('data', (chunk) => {
    data += chunk;
  });

  // The whole response has been received.
  res.on('end', () => {
    console.log('Public IP address:', data);
  });
}).on('error', (err) => {
  console.error('Error retrieving public IP address:', err);
});