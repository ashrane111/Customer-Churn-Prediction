import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up
    { duration: '1m', target: 50 },   // Stay at 50 RPS
    { duration: '30s', target: 100 }, // Spike test
    { duration: '30s', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<150'], // 95% of requests under 150ms
    errors: ['rate<0.02'],             // Error rate under 2%
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8000';

export default function () {
  // Test data
  const payload = JSON.stringify({
    customer_id: `test_${Math.random()}`,
    tenure: Math.floor(Math.random() * 72),
    monthly_charges: 20 + Math.random() * 100,
    total_charges: 100 + Math.random() * 5000,
    contract: ['Month-to-month', 'One year', 'Two year'][Math.floor(Math.random() * 3)],
    payment_method: 'Electronic check',
    internet_service: 'Fiber optic',
    online_security: 'Yes',
    tech_support: 'No'
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  // Make request
  const res = http.post(`${BASE_URL}/predict`, payload, params);

  // Checks
  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'response has probability': (r) => JSON.parse(r.body).churn_probability !== undefined,
    'latency OK': (r) => r.timings.duration < 150,
  });

  errorRate.add(!success);
  sleep(0.1);
}